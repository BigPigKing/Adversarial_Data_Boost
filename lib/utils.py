import torch
import pickle

from tqdm import tqdm
from allennlp.nn.util import move_to_device
from torch.utils.data import DataLoader
from nltk.corpus import wordnet
from typing import Dict, List
from allennlp.data import Vocabulary, DatasetReader, allennlp_collate
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


def unpad_text_field_tensors(
    text_field_tensors: Dict[str, Dict[str, torch.Tensor]],
    is_transformer: bool,
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

    if is_transformer is True:
        target_text_field_tensor = text_field_tensors["tokens"]["token_ids"]
    else:
        target_text_field_tensor = text_field_tensors["tokens"]["tokens"]

    for sent in target_text_field_tensor:
        # sent_len = len(sent) - (sent == padded_idx).sum()

        text_tensor_list.append(sent.clone())

    if len(text_tensor_list) != 1:
        raise ValueError("Augmented but with non-valid batch_size")
    else:
        return text_tensor_list


def pad_text_tensor_list(
    text_tensor_list: List[torch.tensor],
    is_transformer: bool,
    indexer=None,
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

    if is_transformer is True:
        assert indexer, "Use transformer but cannot find indexer!"

        padding_length = len(text_tensor_list[0]["token_ids"])
        text_tensor_dict = indexer.as_padded_tensor_dict(
            text_tensor_list[0],  # Allennlp only accept one element [BUG Alert]!
            padding_lengths={
                "token_ids": padding_length,
                "mask": padding_length,
                "type_ids": padding_length
            }
        )

        for key, value in text_tensor_dict.items():
            text_tensor_dict[key] = value.unsqueeze(0)

        return {"tokens": text_tensor_dict}
    else:
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
    is_transformer: bool,
    is_tokenized=False
):
    # Warning: Trash Code.
    tokenizer = PretrainedTransformerTokenizer(
        "roberta-base"
    )
    if is_transformer is True:
        sentences_token_ids = text_field_tensors["tokens"]["token_ids"].int().tolist()
        return tokenizer.tokenizer.decode(
            sentences_token_ids[0],
            skip_special_tokens=True
        )
    else:
        sentences_token_ids = text_field_tensors["tokens"]["tokens"].int().tolist()

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


def augment_and_get_texts_from_dataset(
    dataset_reader: DatasetReader,
    dataset: AllennlpDataset,
    reinforcer
):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=allennlp_collate)

    augment_texts = []

    for episode_idx, episode in enumerate(tqdm(dataloader)):
        episode = move_to_device(episode, 0)

        # Get augment string from reinforcer
        augment_text = reinforcer.augment(episode["tokens"])

        augment_texts.append(augment_text)

    return augment_texts


def augment_and_get_instances_from_dataset(
    dataset_reader: DatasetReader,
    dataset: AllennlpDataset,
    reinforcer
):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=allennlp_collate)

    augment_instances = []

    for episode_idx, episode in enumerate(tqdm(dataloader)):
        episode = move_to_device(episode, 0)

        # Get augment tokens from reinforcer
        aug_tokens = reinforcer.augment(episode["tokens"])  # Because we only have one sentence in this scenario

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


def new_save_augmentation_sentence(
    policy_weight_paths: List[str],
    saved_names: List[str],
    dataset_reader: DatasetReader,
    train_dataset: AllennlpDataset,
    reinforcer
):
    for policy_weight_path, saved_name in zip(policy_weight_paths, saved_names):
        import time
        start_time = time.time()

        print("Generating augmented instances with {}".format(policy_weight_path))
        # Load pretrained_weight
        reinforcer.policy.load_state_dict(torch.load(policy_weight_path + ".pkl"))

        # Get Augmented Sentence
        augment_texts = augment_and_get_texts_from_dataset(
            dataset_reader,
            train_dataset,
            reinforcer
        )

        # Save obj
        save_obj(augment_texts, saved_name)

        print("--- %s seconds ---" % (time.time() - start_time))

    return


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


def set_augments_to_dataset(
    dataset: AllennlpDataset,
    augmented_instances_save_names: List
):
    for save_name in augmented_instances_save_names:
        augment_texts = load_obj(save_name)

        for instance in dataset.instances:
            instance.add_field


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