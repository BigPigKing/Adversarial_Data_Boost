import torch
import numpy as np

from overrides import overrides
from typing import List, Dict
from .augmenter import Augmenter
from allennlp.data import Vocabulary
from allennlp.nn.util import get_token_ids_from_text_field_tensors, get_text_field_mask
from allennlp.modules.feedforward import FeedForward


class Environment(object):
    def __init__(
        self,
        embedder: torch.nn.Module,
        encoder: torch.nn.Module,
        classifier: torch.nn.Module,
        augmenter_list: List[Augmenter],
        similarity_threshold: float
    ):
        # Module initialization
        self.embedder = embedder
        self.encoder = encoder
        self.classifier = classifier
        self.augmenter_list = augmenter_list

        # Environment Variable
        self.initial_state = None
        self.current_state = None
        self.encoded_initial_state = None
        self.initial_prediction = None
        self.similarity_threshold = similarity_threshold

        # Calculation Function
        self.cos_similarity = torch.nn.CosineSimilarity()
        self.mse_loss_reward = torch.nn.MSELoss()

    def get_current_state(
        self
    ):
        return self.current_state

    def get_encoded_state(
        self,
        state: Dict[str, Dict[str, torch.Tensor]]
    ):
        # Embedded first
        embedded_state = self.embedder(state)

        # get token mask
        state_mask = get_text_field_mask(state)

        # Encode
        encoded_state = self.encoder(embedded_state, state_mask)

        return encoded_state

    def reset(
        self,
        wrapped_token_of_sent: Dict[str, Dict[str, torch.Tensor]]
    ):
        self.initial_state = wrapped_token_of_sent
        self.encoded_initial_state = self.get_encoded_state(self.initial_state)
        self.initial_prediction = self.classifier(self.encoded_initial_state)
        self.current_state = wrapped_token_of_sent

        return self.initial_state

    def _get_env_respond(
        self,
        augmented_state: Dict[str, Dict[str, torch.Tensor]]
    ):
        done = False

        # Get encoded augmented sentence embedding for similarity calculation preparation
        encoded_augmented_state = self.get_encoded_state(augmented_state)

        # Calculate Reward
        if self.cos_similarity(self.encoded_initial_state, encoded_augmented_state) < self.similarity_threshold:
            done = True
            reward = -1.00
        else:
            augmented_prediction = self.classifier(encoded_augmented_state)
            reward = -np.log(self.mse_loss_reward(self.initial_prediction.detach(), augmented_prediction.detach()).item())
            reward = np.clip(reward, 0, 5)

        return reward, done

    def step(
        self,
        action: int
    ):
        done = False
        reward = 0.0

        # Last action will be "stop"
        if action == len(self.augmenter_list) - 1:
            done = True
            reward = 1.0
        else:
            augmented_state = self.augmenter_list[action].augment(self.current_state)
            reward, done = self._get_env_respond(augmented_state)

            # move to next state
            self.current_state = augmented_state

        return self.current_state, reward, done


class Policy(torch.nn.Module):
    def __init__(
        self,
        embedder: torch.nn.Module,
        encoder: torch.nn.Module,
        input_dim: int,
        num_of_action: int
    ):
        super(Policy, self).__init__()

        self.embedder = embedder
        self.encoder = encoder

        self.feedforward = FeedForward(
            input_dim,
            3,
            [64, 32, num_of_action],
            torch.nn.ReLU()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

    def select_action(
        self,
        state: Dict[str, Dict[str, torch.Tensor]]
    ):
        action, action_probs = self(state)

        return action, action_probs

    @overrides
    def forward(
        self,
        state: Dict[str, Dict[str, torch.Tensor]]
    ):
        # Embedded first
        embedded_state = self.embedder(state)

        # get token mask
        state_mask = get_text_field_mask(state)

        # Encode
        encoded_state = self.encoder(embedded_state, state_mask)

        # Get action probs
        # encoded_state = torch.ones(encoded_state.shape).to(0)
        # print(encoded_state)
        action_scores = self.feedforward(encoded_state.detach())
        action_probs = torch.nn.functional.softmax(action_scores)

        # Get action
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()

        return action.item(), action_probs


class REINFORCER(torch.nn.Module):
    def __init__(
        self,
        embedder: torch.nn.Module,
        encoder: torch.nn.Module,
        classifier: torch.nn.Module,
        augmenter_list: List[Augmenter],
        vocab: Vocabulary,
        max_step: int = 9,
        gamma: float = 0.999
    ):
        super(REINFORCER, self).__init__()

        self.policy = Policy(
            embedder,
            encoder,
            encoder.get_output_dim(),
            len(augmenter_list)
        )
        self.env = Environment(
            embedder,
            encoder,
            classifier,
            augmenter_list,
            0.4
        )

        self.vocab = vocab

        self.max_step = max_step
        self.gamma = gamma

    def _get_token_of_sents(
        self,
        token_ids: Dict[str, Dict[str, torch.Tensor]]
    ):
        token_of_sents = get_token_ids_from_text_field_tensors(token_ids)

        return token_of_sents

    def _wrap_token_of_sent(
        self,
        token_of_sent: torch.Tensor
    ):
        wrapped_token_of_sent = torch.stack([token_of_sent])

        return {"tokens": {"tokens": wrapped_token_of_sent}}

    def _calculate_loss(
        self,
        log_probs,
        rewards
    ):
        R = 0.0
        losses = []
        returns = []

        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())

        for log_prob, R in zip(log_probs, returns):
            losses.append(-log_prob * R)

        loss = torch.cat(losses).sum()

        return loss

    def optimize(
        self,
        loss
    ):
        print(loss)
        loss.backward()
        self.policy.optimizer.step()
        self.policy.optimizer.zero_grad()

    def _print_tokens_from_index(
        self,
        token_ids: torch.Tensor
    ):
        token_ids = token_ids.detach().clone().tolist()[0]
        tokens = []

        for token_id in token_ids:
            tokens.append(self.vocab.get_token_from_index(token_id))

        print(tokens)

    def _report_action(
        self,
        step: int,
        action: int,
        origin_state: Dict[str, Dict[str, torch.Tensor]],
        aug_state: Dict[str, Dict[str, torch.Tensor]]
    ):
        if step == 0:
            print("Origin Token: ")
            self._print_tokens_from_index(origin_state["tokens"]["tokens"])
        print("Step        : " + str(step))
        print("Action      : " + str(action))
        print("Aug Token   : ")
        self._print_tokens_from_index(aug_state["tokens"]["tokens"])

    @overrides
    def forward(
        self,
        wrapped_token_of_sent: Dict[str, Dict[str, torch.Tensor]]
    ):
        """
        forward: get loss(output dict) from currenet episode(currenet sentence)
        """
        output_dict = {}

        state = self.env.reset(wrapped_token_of_sent)
        log_probs = []
        rewards = []

        for step in range(self.max_step):
            action, action_log_prob = self.policy.select_action(state)
            state, reward, done = self.env.step(action)

            log_probs.append(action_log_prob)
            rewards.append(reward)

            self._report_action(step, action, wrapped_token_of_sent, state)

            if done:
                break

        # calculate loss
        print(log_probs)
        print(rewards)
        loss = self._calculate_loss(log_probs, rewards)

        # Prepare output dict
        output_dict["loss"] = loss

        return output_dict


def main():
    pass


if __name__ == '__main__':
    main()
