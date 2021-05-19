import torch
import pickle

from allennlp.nn.util import move_to_device
from torch.utils.data import DataLoader
from nltk.corpus import wordnet
from typing import Dict, List
from allennlp.data import Vocabulary, DatasetReader, allennlp_collate
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset


def unpad_text_field_tensors(
    text_field_tensors: Dict[str, Dict[str, torch.Tensor]],
    padded_idx: int = 0
) -> List[torch.Tensor]:
    """
    Unpad text field tensors and return a list which lenth is the batch size of text_field_tensor.

    Example:
        Input:
            text_field_tensors = {"tokens": {"tokens": tensor with shape: (B, S)}}
            B = num_of_batch, S = padded sequence
        Output:
            List[tensor with shape (M) * B]
            B = num_of_batch, M = unpadded sequence lenth for different sentence
    """
    text_tensor_list = []
    target_text_field_tensor = text_field_tensors["tokens"]["tokens"]

    for sent in target_text_field_tensor:
        sent_len = len(sent) - (sent == padded_idx).sum()

        text_tensor_list.append(sent[:sent_len].clone())

    return text_tensor_list


def pad_text_tensor_list(
    text_tensor_list: List[torch.tensor],
    padded_idx: int = 0
):
    """
    Pad list of text tensor back to text_field_tensors with type {Dict[str, Dict[str, torch.Tensor]]}.

    Example:
        Input:
            List[tensor with shape (M) * B]
            B = num_of_batch, M = unpadded sequence lenth for different sentence
        Output:
            text_field_tensors = {"tokens": {"tokens": tensor with shape: (B, S)}}
            B = num_of_batch, S = padded sequence

    """
    padded_text_tensor = torch.nn.utils.rnn.pad_sequence(text_tensor_list, batch_first=True)

    return {"tokens": {"tokens": padded_text_tensor}}


def add_wordnet_to_vocab(
    vocab: Vocabulary
):
    # iterate over all the possible synomys to enrich vocabulary set
    for syn in wordnet.all_synsets():
        for synonym_lemma in syn.lemmas():
            synonym = synonym_lemma.name().replace('_', ' ').replace('-', ' ').lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])

            for synonym_token in synonym.split():
                vocab.add_token_to_namespace(synonym_token)

    return vocab


def get_sentence_from_text_field_tensors(
    vocab: Vocabulary,
    text_field_tensors: Dict[str, Dict[str, torch.Tensor]],
    default_field_name: str = "tokens",
    is_tokenized=False
):
    sentences_token_ids = text_field_tensors["tokens"][default_field_name].int().tolist()
    sentences = []

    for sentence_token_ids in sentences_token_ids:
        sentence_tokens = []

        for sentence_token_id in sentence_token_ids:
            sentence_tokens.append(vocab.get_token_from_index(sentence_token_id))

        if is_tokenized is False:
            sentences.append(' '.join(sentence_tokens))
        else:
            sentences.append(sentence_tokens)

    return sentences


def augment_and_get_instances_from_dataset(
    dataset_reader: DatasetReader,
    dataset: AllennlpDataset,
    reinforcer
):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=allennlp_collate)

    augment_instances = []

    for episode_idx, episode in enumerate(dataloader):
        episode = move_to_device(episode, 0)

        # Get augment tokens from reinforcer
        reinforcer.eval()
        aug_tokens = reinforcer.augment(episode["tokens"])[-1]  # Because we only have one sentence in this scenario

        # text to instance
        augment_instance = dataset_reader.text_to_instance(
            aug_tokens,
            reinforcer.vocab.get_token_from_index(
                episode["label"].int().item(),
                namespace="labels"
            ),
            augment=reinforcer.env.similarity_threshold,
            is_augment=True
        )
        augment_instances.append(augment_instance)

    # return dataset
    return augment_instances


def get_synonyms_from_dataset(
    dataset: AllennlpDataset
):
    synonym_dict = {}

    for instance in dataset.instances:
        for true_token in instance["tokens"]:
            token = str(true_token)
            synonyms = set()

            for syn in wordnet.synsets(token):
                for synonym_lemma in syn.lemmas():
                    synonym = synonym_lemma.name().replace('_', ' ').replace('-', ' ').lower()
                    synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                    synonyms.add(tuple(synonym.split()))

            synonyms = list(synonyms)
            synonym_dict[token] = synonyms

    return synonym_dict


def get_and_save_augmentation_sentence(
    policy_weight_paths: List[str],
    saved_names: List[str],
    dataset_reader: DatasetReader,
    train_dataset: AllennlpDataset,
    reinforcer
):
    total_augment_instances = []

    for policy_weight_path, saved_name in zip(policy_weight_paths, saved_names):
        import time
        start_time = time.time()

        print("Generating augmented instances with {}".format(policy_weight_path))
        # Load pretrained_weight
        reinforcer.policy.load_state_dict(torch.load(policy_weight_path + ".pkl"))

        # Get Augmented Sentence
        augment_instances = augment_and_get_instances_from_dataset(
            dataset_reader,
            train_dataset,
            reinforcer
        )

        # Save obj
        save_obj(augment_instances, saved_name)

        # Collect augmented instances
        total_augment_instances += augment_instances
        print("--- %s seconds ---" % (time.time() - start_time))

    return total_augment_instances


def save_obj(
    obj: object,
    name: str
):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(
    name: str
):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def main():
    pass


if __name__ == '__main__':
    main()
