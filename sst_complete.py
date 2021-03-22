import json
import torch
import logging

from typing import Dict, List

from lib.dataset import get_sst_ds
from lib.embedder import TextEmbedder
from lib.encoder import TextEncoder
from lib.classifier import TextClassifier
from lib.model import SentimentModel
from lib.trainer import TextTrainer, ReinforceTrainer
from lib.augmenter import Augmenter, DeleteAugmenter, SwapAugmenter
from lib.augmenter import IdentityAugmenter, InsertAugmenter, ReplaceAugmenter
from lib.reinforcer import REINFORCER
from lib.utils import add_wordnet_to_vocab, get_synonyms_from_dataset, load_obj

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from allennlp.data import allennlp_collate
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.modules.token_embedders import Embedding
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy


logger = logging.getLogger(__name__)


def get_config_params(
    config_filepath: str
):
    with open(config_filepath) as f:
        config_params = json.load(f)

    return config_params


def set_and_get_dataset(
    dataset_params: Dict
):
    if dataset_params["select_dataset"] == "sst":
        train_ds, valid_ds, test_ds, dataset_reader = get_sst_ds(
            granularity=dataset_params["sst"]["granularity"]
        )

        return train_ds, valid_ds, test_ds, dataset_reader
    else:
        raise(KeyError)


def set_and_get_dataloader(
    dataloader_params: Dict,
    train_ds: AllennlpDataset,
    valid_ds: AllennlpDataset,
    test_ds: AllennlpDataset
):
    train_data_loader = DataLoader(
        train_ds,
        batch_size=dataloader_params["batch_size"],
        shuffle=dataloader_params["shuffle"],
        collate_fn=allennlp_collate
    )

    valid_data_loader = DataLoader(
        valid_ds,
        batch_size=dataloader_params["batch_size"],
        collate_fn=allennlp_collate
    )

    test_data_loader = DataLoader(
        test_ds,
        batch_size=dataloader_params["batch_size"],
        collate_fn=allennlp_collate
    )

    return train_data_loader, valid_data_loader, test_data_loader


def set_and_get_embedder(
    embedder_params: Dict,
    vocab: Vocabulary
):
    if embedder_params["pretrained_embedding"] == "glove":
        pretrained_embedding = Embedding(
            embedding_dim=embedder_params["glove"]["embedding_dim"],
            vocab=vocab,
            padding_index=embedder_params["glove"]["padding_index"],
            pretrained_file=embedder_params["glove"]["pretrained_filepath"]
        )

        return TextEmbedder(pretrained_embedding=pretrained_embedding)
    else:
        raise(KeyError)


def set_and_get_encoder(
    encoder_params: Dict,
    input_size: int
):
    encoder = TextEncoder(
        input_size=input_size,
        s2s_encoder_params=encoder_params["s2s_encoder"],
        s2v_encoder_params=encoder_params["s2v_encoder"]
    )

    if encoder_params["pretrained_weight"] != "none":
        encoder.load_state_dict(
            torch.load(
                encoder_params["pretrained_weight"]
            )
        )
    else:
        pass

    return encoder


def set_and_get_classifier(
    classifier_params: Dict,
    input_size: int,
    output_size: int
):
    # Instancelize ReLU
    if classifier_params["feedforward"]["activations"] == "relu":
        classifier_params["feedforward"]["activations"] = torch.nn.ReLU()
    else:
        raise(KeyError)

    classifier = TextClassifier(
        input_size=input_size,
        output_size=output_size,
        feedforward_params=classifier_params["feedforward"]
    )

    return classifier


def set_and_get_synonyms(
    synonyms_params: Dict,
    train_ds: AllennlpDataset
):
    if synonyms_params["synonyms_filepath"] == "sst_synonyms":
        synonyms = load_obj("sst_synonyms")
        return synonyms
    else:
        synonyms = get_synonyms_from_dataset(train_ds)
        return synonyms


def set_and_get_augmenters(
    augmenter_params: Dict,
    vocab: Vocabulary,
    synonyms: Dict
):
    # Delete
    delete_augmenter = DeleteAugmenter(
        augmenter_params["delete_augmenter"]
    )

    # Swap
    swap_augmenter = SwapAugmenter(
        augmenter_params["swap_augmenter"]
    )

    # Replace
    replace_augmenter = ReplaceAugmenter(
        vocab,
        synonyms,
        augmenter_params["replace_augmenter"]
    )

    # Insert
    insert_augmenter = InsertAugmenter(
        vocab,
        synonyms,
        augmenter_params["insert_augmenter"]
    )

    # Identity
    identity_augmenter = IdentityAugmenter()

    augmenters = [
        delete_augmenter,
        swap_augmenter,
        replace_augmenter,
        insert_augmenter,
        identity_augmenter
    ]

    returned_augmenters = []

    for idx in augmenter_params["select_idx"]:
        returned_augmenters.append(augmenters[idx])

    return returned_augmenters


def set_and_get_reinforcer(
    reinforcer_params: Dict,
    embedder: TextEmbedder,
    encoder: TextEncoder,
    classifier: TextClassifier,
    augmenters: List[Augmenter],
    vocab: Vocabulary
):
    # Instanize ReLU
    if reinforcer_params["policy"]["feedforward"]["activations"] == "relu":
        reinforcer_params["policy"]["feedforward"]["activations"] = torch.nn.ReLU()
    else:
        raise(KeyError)

    # Complete hidden dimsions
    reinforcer_params["policy"]["feedforward"]["hidden_dims"].append(len(augmenters))

    # Declare Optimizer
    if reinforcer_params["policy"]["optimizer"]["select_optimizer"] == "adam":
        reinforcer_params["policy"]["optimizer"]["select_optimizer"] = torch.optim.Adam
    else:
        raise(KeyError)

    reinforcer = REINFORCER(
        embedder,
        encoder,
        classifier,
        augmenters,
        vocab,
        reinforcer_params["env"],
        reinforcer_params["policy"],
        reinforcer_params["REINFORCE"]
    )

    return reinforcer


def set_and_get_sentiment_model(
    sentiment_model_params: Dict,
    vocab: Vocabulary,
    embedder: TextEmbedder,
    encoder: TextEncoder,
    classifier: TextClassifier,
    reinforcer: REINFORCER
):
    # Instanize Criterions
    sentiment_model_params["criterions"]["classification_criterion"] = torch.nn.CrossEntropyLoss()
    sentiment_model_params["criterions"]["contrastive_criterion"] = torch.nn.CosineEmbeddingLoss()

    # Instanize Evaluation
    if sentiment_model_params["evaluation"] == "categorical":
        sentiment_model_params["evaluation"] = CategoricalAccuracy()
    else:
        raise(KeyError)

    # Optimize Declartion
    if sentiment_model_params["optimizer"]["select_optimizer"] == "adam":
        sentiment_model_params["optimizer"]["select_optimizer"] = torch.optim.Adam
    else:
        raise(KeyError)

    sentiment_model = SentimentModel(
        vocab,
        embedder,
        encoder,
        classifier,
        reinforcer,
        sentiment_model_params
    )

    return sentiment_model


def set_and_get_text_trainer(
    text_trainer_params: Dict,
    sentiment_model: SentimentModel
):
    text_trainer = TextTrainer(
        sentiment_model,
        is_save=text_trainer_params["is_save"]
    )

    return text_trainer


def set_and_get_reinforce_trainer(
    reinforce_trainer_params: Dict,
    reinforcer: REINFORCER,
    writer: SummaryWriter
):
    reinforce_trainer = ReinforceTrainer(
        reinforcer,
        writer,
        is_save=reinforce_trainer_params["is_save"]
    )

    return reinforce_trainer


def main(config_params):
    # Get Dataset
    train_ds, valid_ds, test_ds, dataset_reader = set_and_get_dataset(
        config_params["dataset"]
    )

    # Set Vocabulary Set
    vocab = Vocabulary.from_instances(train_ds)
    vocab = add_wordnet_to_vocab(vocab)
    train_ds.index_with(vocab)
    valid_ds.index_with(vocab)
    test_ds.index_with(vocab)

    # Get Dataloader
    train_data_loader, valid_data_loader, test_data_loader = set_and_get_dataloader(
        config_params["dataloader"], train_ds, valid_ds, test_ds
    )

    # Get Embedder
    embedder = set_and_get_embedder(
        config_params["embedder"],
        vocab
    )

    # Get Encoder
    encoder = set_and_get_encoder(
        config_params["encoder"],
        embedder.get_output_dim()
    )

    # Get Classifier
    classifier = set_and_get_classifier(
        config_params["classifier"],
        encoder.get_output_dim(),
        vocab.get_vocab_size("labels")
    )

    # Schedular declartion
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

    # Get Synonyms
    synonyms = set_and_get_synonyms(
        config_params["synonyms"],
        train_ds
    )

    # Get Augmenters
    augmenters = set_and_get_augmenters(
        config_params["augmenter"],
        vocab,
        synonyms
    )

    # Get Reinforcer
    reinforcer = set_and_get_reinforcer(
        config_params["reinforcer"],
        embedder,
        encoder,
        classifier,
        augmenters,
        vocab
    )

    # Get Sentiment Model
    sentiment_model = set_and_get_sentiment_model(
        config_params["sentiment_model"],
        vocab,
        embedder,
        encoder,
        classifier,
        reinforcer
    )

    # Set Writer
    writer = SummaryWriter()

    # Set GPU
    if config_params["env"]["USE_GPU"] is not None:
        print("Hello")
        sentiment_model = sentiment_model.cuda()

    # Get Text Trainer
    text_trainer = set_and_get_text_trainer(
        config_params["trainer"]["text_trainer"],
        sentiment_model
    )

    # Get Reinforce Trainer
    reinforce_trainer = set_and_get_reinforce_trainer(
        config_params["trainer"]["reinforce_trainer"],
        reinforcer,
        writer
    )

    # Sentiment Model Train
    text_trainer.fit(
        config_params["trainer"]["text_trainer"]["epochs"],
        train_data_loader,
        valid_data_loader,
        test_data_loader
    )

    # Reinforce Model Train
    train_data_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        collate_fn=allennlp_collate
    )

    reinforce_trainer.fit(
        config_params["trainer"]["reinforce_trainer"]["epochs"],
        config_params["trainer"]["reinforce_trainer"]["batch_size"],
        train_data_loader
    )


if __name__ == '__main__':
    main(get_config_params("model_config.json"))
