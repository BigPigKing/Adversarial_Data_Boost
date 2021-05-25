import torch
import numpy as np

from overrides import overrides
from typing import List, Dict
from .utils import get_sentence_from_text_field_tensors
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
        max_step: int,
        env_params: Dict
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
        self.current_step = None
        self.max_step = max_step
        self.similarity_threshold = env_params["similarity_threshold"]

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
        with torch.no_grad():
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
        with torch.no_grad():
            self.initial_prediction = self.classifier(self.encoded_initial_state)
        self.current_state = wrapped_token_of_sent
        self.current_step = 0

        return self.initial_state

    def _get_env_respond(
        self,
        augmented_state: Dict[str, Dict[str, torch.Tensor]]
    ):
        done = False

        # Get encoded augmented sentence embedding for similarity calculation preparation
        encoded_augmented_state = self.get_encoded_state(augmented_state)

        # Calculate Reward - Typical
        if self.cos_similarity(self.encoded_initial_state, encoded_augmented_state) < self.similarity_threshold:
            done = True
            reward = -0.6
        else:
            with torch.no_grad():
                augmented_prediction = self.classifier(encoded_augmented_state)
            reward = np.log(
                self.mse_loss_reward(
                    self.initial_prediction.detach(),
                    augmented_prediction.detach()
                ).cpu().item()
            )
            reward = np.clip(reward, -20, -1)
            reward = 1 / - reward

        # Penelty Reward
        penelty_reward = 0.6 * self.current_step / self.max_step

        # Record Step
        self.current_step += 1

        return reward - penelty_reward, done

    def step(
        self,
        action: int
    ):
        done = False
        reward = 0.0

        # Last action will be "stop"
        if action == len(self.augmenter_list) - 1:
            done = True
            reward = 1
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
        num_of_action: int,
        policy_params
    ):
        super(Policy, self).__init__()

        self.embedder = embedder
        self.encoder = encoder

        self.feedforward = FeedForward(
            input_dim,
            num_layers=policy_params["feedforward"]["num_layers"],
            hidden_dims=policy_params["feedforward"]["hidden_dims"],
            activations=policy_params["feedforward"]["activations"],
            dropout=policy_params["feedforward"]["dropout"]
        )

        self.optimizer = policy_params["optimizer"]["select_optimizer"](
            self.parameters(),
            lr=policy_params["optimizer"]["lr"])

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
        with torch.no_grad():
            # Embedded first
            embedded_state = self.embedder(state)

            # get token mask
            state_mask = get_text_field_mask(state)

            # Encode
            encoded_state = self.encoder(embedded_state, state_mask)

        # Get action probs
        action_scores = self.feedforward(encoded_state.detach())
        action_probs = torch.nn.functional.softmax(action_scores, dim=1)

        # Get action
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()

        return action.item(), m.log_prob(action)


class REINFORCER(torch.nn.Module):
    def __init__(
        self,
        embedder: torch.nn.Module,
        encoder: torch.nn.Module,
        classifier: torch.nn.Module,
        augmenter_list: List[Augmenter],
        vocab: Vocabulary,
        env_params: Dict,
        policy_params: Dict,
        REINFORCE_params: Dict
    ):
        super(REINFORCER, self).__init__()

        embedder.eval()
        encoder.eval()
        classifier.eval()

        self.policy = Policy(
            embedder,
            encoder,
            encoder.get_output_dim(),
            len(augmenter_list),
            policy_params
        )
        self.env = Environment(
            embedder,
            encoder,
            classifier,
            augmenter_list,
            REINFORCE_params["max_step"],
            env_params
        )

        self.vocab = vocab

        self.max_step = REINFORCE_params["max_step"]
        self.gamma = REINFORCE_params["gamma"]
        self.clip_grad = REINFORCE_params["clip_grad"]

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

        # Calculate cumulated reward
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)

        # If the length equals to one, we do not need to do any standardlization
        if len(rewards) == 1:
            pass
        else:
            returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())

        # Calculate loss
        for log_prob, R in zip(log_probs, returns):
            losses.append(-log_prob * R)

        loss = torch.cat(losses).sum()

        return loss

    def optimize(
        self,
        loss
    ):
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)

        self.policy.optimizer.step()
        self.policy.optimizer.zero_grad()

    def augment(
        self,
        wrapped_token_of_sent: Dict[str, Dict[str, torch.Tensor]]
    ):
        state = self.env.reset(wrapped_token_of_sent)
        log_probs = []
        rewards = []

        for step in range(self.max_step):
            action, action_log_prob = self.policy.select_action(state)
            state, reward, done = self.env.step(action)

            log_probs.append(action_log_prob)
            rewards.append(reward)

            if done is True:
                break

        # Prepare output dict
        return get_sentence_from_text_field_tensors(self.vocab, state, is_tokenized=True)

    @overrides
    def forward(
        self,
        episode: Dict[str, torch.Tensor]
    ):
        """
        forward: get loss(output dict) from currenet episode(currenet sentence)
        """
        output_dict = {}

        wrapped_token_of_sent = episode["tokens"]

        state = self.env.reset(wrapped_token_of_sent)
        log_probs = []
        rewards = []
        actions = []

        for step in range(self.max_step):
            action, action_log_prob = self.policy.select_action(state)
            state, reward, done = self.env.step(action)

            log_probs.append(action_log_prob)
            rewards.append(reward)
            actions.append(action)

            if done is True:
                break

        # calculate loss
        loss = self._calculate_loss(log_probs, rewards)

        # Prepare output dict
        output_dict["origin_sentence"] = get_sentence_from_text_field_tensors(self.vocab, wrapped_token_of_sent)
        output_dict["augment_sentence"] = get_sentence_from_text_field_tensors(self.vocab, state)
        output_dict["actions"] = actions
        output_dict["loss"] = loss
        output_dict["ep_reward"] = torch.sum(torch.tensor(rewards), dim=0)
        # print(output_dict["ep_reward"])

        return output_dict


def main():
    pass


if __name__ == '__main__':
    main()
