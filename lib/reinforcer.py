import torch

from overrides import overrides
from typing import List, Dict
from .augmenter import Augmenter
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

    def get_encoded_state_from_token_of_sent(
        self,
        wrapped_token_of_sent: Dict[str, Dict[str, torch.Tensor]]
    ):
        # Embedded first
        embedded_sent = self.embedder(wrapped_token_of_sent)

        # get token mask
        sent_mask = get_text_field_mask(wrapped_token_of_sent)

        # Encode
        encoded_sent_state = self.encoder(embedded_sent.detach(), sent_mask)

        return encoded_sent_state

    def reset(
        self,
        wrapped_token_of_sent: Dict[str, Dict[str, torch.Tensor]]
    ):
        self.initial_state = wrapped_token_of_sent
        self.encoded_initial_state = self.get_encoded_state_from_token_of_sent(self.initial_state)
        self.initial_prediction = self.classifier(self.encoded_initial_state.detach())
        self.current_state = wrapped_token_of_sent

        return wrapped_token_of_sent

    def _get_env_respond(
        self,
        augmented_sent: Dict[str, Dict[str, torch.Tensor]]
    ):
        done = False

        # Get encoded augmented sentence embedding for similarity calculation preparation
        encoded_augmented_state = self.get_encoded_state_from_token_of_sent(augmented_sent)

        # Calculate Reward
        if self.cos_similarity(self.encoded_initial_state, self.encoded_augmented_state) < self.similarity_threshold:
            done = True
            reward = -1.00
        else:
            augmented_prediction = self.classifier(encoded_augmented_state.detach())
            reward = self.mse_loss_reward(self.initial_prediction, augmented_prediction).item()

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
            reward = 0.0
        else:
            augmented_sent = self.augmenter_list[action].augment(self.current_state)
            reward, done = self._get_env_respond(augmented_sent)

            # move to next state
            self.current_state = augmented_sent

        return self.current_state, reward, done


class Policy(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_of_action: int
    ):
        super(Policy, self).__init__()

        self.feedforward = FeedForward(
            input_dim,
            3,
            [64, 32, num_of_action],
            torch.nn.ReLU()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    @overrides
    def forward(
        self,
        encoded_sent_state: torch.Tensor
    ):
        action_scores = self.feedforward(encoded_sent_state)

        return torch.nn.functional.softmax(action_scores, dim=1)


class REINFORCER(torch.nn.Module):
    def __init__(
        self,
        embedder: torch.nn.Module,
        encoder: torch.nn.Module,
        classifier: torch.nn.Module,
        augmenter_list: List[Augmenter],
        max_step: int = 3,
        gamma: float = 0.99
    ):
        super(REINFORCER, self).__init__()

        self.policy = Policy(
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

        print(wrapped_token_of_sent)
        print(wrapped_token_of_sent.shape)

        return {"tokens": {"tokens": wrapped_token_of_sent}}

    def _calculate_loss(
        self,
        entropies,
        log_probs,
        rewards
    ):
        R = torch.zeros(1, 1)
        loss = 0.0

        for i in reversed(range(len(rewards))):
            R = self.gamma * R + rewards[i]
            loss = loss - (log_probs[i]*(torch.autograd.Variable(R).expand_as(log_probs[i])).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()

        loss = loss / len(rewards)

        return loss

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
        entropies = []
        log_probs = []
        rewards = []

        for step in self.max_step:
            action, action_log_prob, action_entropy = self.policy.select_action(state)
            state, reward, done = self.env.step(action)

            entropies.append(action_entropy)
            log_probs.append(action_log_prob)
            rewards.append(reward)

            if done:
                break

        # calculate loss
        loss = self._calculate_loss(entropies, log_probs, rewards)

        # Prepare output dict
        output_dict["loss"] = loss

        return output_dict


def main():
    pass


if __name__ == '__main__':
    main()
