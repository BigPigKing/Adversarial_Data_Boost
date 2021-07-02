import json
import torch
import torch.multiprocessing as mp

from typing import Dict, List
from itertools import repeat

from lib.dataset import get_sst_ds, get_yelp_ds
from lib.embedder import TextEmbedder
from lib.encoder import WordEncoder, TransformerEncoder
from lib.classifier import TextClassifier
from lib.model import SentimentModel
from lib.trainer import TextTrainer, ReinforceTrainer
from lib.augmenter import DeleteAugmenter, SwapAugmenter
from lib.augmenter import IdentityAugmenter, InsertAugmenter, ReplaceAugmenter
from lib.reinforcer import REINFORCER
from lib.utils import load_obj, generate_and_save_augmentation_texts
from lib.visualizer import TSNEVisualizer, IsomapVisualizer

from torch.utils.data import DataLoader
from allennlp.data import allennlp_collate
from allennlp.data.fields import TextField
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset, DatasetReader
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerEmbedder
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy


def get_config_params(
    config_filepath: str
) -> Dict:
    with open(config_filepath) as f:
        config_params = json.load(f)

    return config_params


def set_and_get_dataset(
    dataset_params: Dict
) -> Dict:
    if dataset_params["select_dataset"] == "sst":
        train_ds, valid_ds, test_ds, dataset_reader = get_sst_ds(
            dataset_params["sst"]
        )

        return {
            "train_ds": train_ds,
            "valid_ds": valid_ds,
            "test_ds": test_ds,
            "dataset_reader": dataset_reader
        }
    elif dataset_params["select_dataset"] == "yelp":
        train_ds, valid_ds, test_ds, dataset_reader = get_yelp_ds(
            dataset_params["yelp"]
        )

        return {
            "train_ds": train_ds,
            "valid_ds": valid_ds,
            "test_ds": test_ds,
            "dataset_reader": dataset_reader
        }

    else:
        raise(KeyError)


def set_and_get_vocab(
    dataset_dict: Dict
) -> Vocabulary:
    vocab = Vocabulary.from_instances(dataset_dict["train_ds"])

    # Index Dataset
    dataset_dict["train_ds"].index_with(vocab)
    dataset_dict["valid_ds"].index_with(vocab)
    dataset_dict["test_ds"].index_with(vocab)

    return vocab


def set_and_get_word_embedder(
    word_embedder_params: Dict,
    vocab: Vocabulary
):
    if word_embedder_params["pretrained_embedding"]["pretrained_filepath"]:
        print("Loading pretrained embedding for word_embedder ...")
        pretrained_embedding = Embedding(
            embedding_dim=word_embedder_params["pretrained_embedding"]["embedding_dim"],
            pretrained_file=word_embedder_params["pretrained_embedding"]["pretrained_filepath"],
            vocab=vocab,
            trainable=word_embedder_params["trainable"]["trainable"],
            projection_dim=word_embedder_params["trainable"]["projection_dim"]
        )
    else:
        print("Random initialize embedding for word_embedder ...")
        pretrained_embedding = Embedding(
            embedding_dim=word_embedder_params["pretrained_embedding"]["embedding_dim"],
            vocab=vocab,
            trainable=word_embedder_params["trainable"]["trainable"],
            projection_dim=word_embedder_params["trainable"]["projection_dim"]
        )

    token_indexers = {"tokens": pretrained_embedding}
    word_embedder = TextEmbedder(token_indexers)

    return word_embedder


def set_and_get_transformer_embedder(
    transformer_embedder_params: Dict
):
    print(
        "Loading transformer pretrained embedding from \"{}\" ...".format(
            transformer_embedder_params["model_name"]
        )
    )
    pretrained_embedding = PretrainedTransformerEmbedder(
        model_name=transformer_embedder_params["model_name"],
        train_parameters=transformer_embedder_params["train_parameters"]
    )

    token_indexers = {"tokens": pretrained_embedding}
    transformer_embedder = TextEmbedder(token_indexers)

    return transformer_embedder


def set_and_get_embedder(
    embedder_params: Dict,
    vocab: Vocabulary
):
    if embedder_params["select_embedder"] == "word_embedder":
        embedder = set_and_get_word_embedder(
            embedder_params["word_embedder"],
            vocab
        )
    elif embedder_params["select_embedder"] == "transformer_embedder":
        embedder = set_and_get_transformer_embedder(
            embedder_params["transformer_embedder"]
        )
    else:
        raise KeyError("Unknowm embedder's name!")

    return embedder


def set_and_get_encoder(
    encoder_params: Dict,
    embedder
):
    if encoder_params["select_encoder"] == "word_encoder":
        encoder = WordEncoder(
            input_size=embedder.get_output_dim(),
            s2s_encoder_params=encoder_params["word_encoder"]["s2s_encoder"],
            s2v_encoder_params=encoder_params["word_encoder"]["s2v_encoder"]
        )
    elif encoder_params["select_encoder"] == "transformer_encoder":
        encoder = TransformerEncoder(
            encoder_params["transformer_encoder"]
        )
    else:
        raise KeyError("Unknown encoder name!")

    return encoder


def set_and_get_text_model(
    text_model_params: Dict,
    dataset_dict: Dict,
    vocab: Vocabulary
) -> torch.nn.Module:
    # Set and get embedder
    embedder = set_and_get_embedder(
        text_model_params["embedder"],
        vocab
    )

    # Set and get encoder
    encoder = set_and_get_encoder(
        text_model_params["encoder"],
        embedder
    )

    # Set and get classifier
    # Instancelize activation function
    if text_model_params["classifier"]["feedforward"]["activations"] == "relu":
        text_model_params["classifier"]["feedforward"]["activations"] = torch.nn.ReLU()
    elif text_model_params["classifier"]["feedforward"]["activations"] == "silu":
        text_model_params["classifier"]["feedforward"]["activations"] = torch.nn.SiLU()
    else:
        raise(KeyError)

    classifier = TextClassifier(
        input_size=encoder.get_output_dim(),
        output_size=vocab.get_vocab_size("labels"),
        feedforward_params=text_model_params["classifier"]["feedforward"]
    )

    # Set and get text model
    # Instanize Criterions
    text_model_params["criterions"]["classification_criterion"] = torch.nn.CrossEntropyLoss()
    text_model_params["criterions"]["contrastive_criterion"] = torch.nn.CosineEmbeddingLoss()

    # Instanize Evaluation
    if text_model_params["evaluation"] == "categorical":
        text_model_params["evaluation"] = CategoricalAccuracy()
    else:
        raise(KeyError)

    # Optimize Declartion
    if text_model_params["optimizer"]["select_optimizer"] == "adam":
        text_model_params["optimizer"]["select_optimizer"] = torch.optim.Adam
    elif text_model_params["optimizer"]["select_optimizer"] == "rmsprop":
        text_model_params["optimizer"]["select_optimizer"] = torch.optim.RMSprop
    elif text_model_params["optimizer"]["select_optimizer"] == "adamw":
        from transformers import AdamW
        text_model_params["optimizer"]["select_optimizer"] = AdamW
    else:
        raise(KeyError)

    text_model = SentimentModel(
        vocab,
        embedder,
        encoder,
        classifier,
        text_model_params,
        dataset_dict
    )

    return text_model


def set_and_get_reinforcer(
    reinforcer_params: Dict,
    dataset_dict: Dict,
    vocab: Vocabulary,
    text_model: torch.nn.Module,
):
    # Get synonyms
    # if reinforcer_params["augmenter"]["synonyms"]["synonyms_filepath"]:
    #     synonyms = load_obj(
    #         reinforcer_params["augmenter"]["synonyms"]["synonyms_filepath"]
    #     )
    # else:
    #     synonyms = get_synonyms_from_dataset(dataset_dict["train_ds"])

    # print(vocab)
    # exit()

    # Get augmenter
    # Delete
    delete_augmenter = DeleteAugmenter(
        reinforcer_params["augmenter"]["delete_augmenter"],
        vocab,
        dataset_dict
    )

    # Swap
    swap_augmenter = SwapAugmenter(
        reinforcer_params["augmenter"]["swap_augmenter"],
        vocab,
        dataset_dict
    )

    # Replace
    replace_augmenter = ReplaceAugmenter(
        reinforcer_params["augmenter"]["replace_augmenter"],
        vocab,
        dataset_dict
    )

    # Insert
    insert_augmenter = InsertAugmenter(
        reinforcer_params["augmenter"]["insert_augmenter"],
        vocab,
        dataset_dict
    )

    # Identity
    identity_augmenter = IdentityAugmenter(dataset_dict["dataset_reader"].is_transformer)

    augmenters = [
        delete_augmenter,
        swap_augmenter,
        replace_augmenter,
        insert_augmenter,
    ]

    # Get selected augmenters
    select_augmenters = []

    for idx in reinforcer_params["augmenter"]["select_augmenter"]:
        select_augmenters.append(augmenters[idx])

    select_augmenters.append(identity_augmenter)

    # Instanize ReLU
    if reinforcer_params["policy"]["feedforward"]["activations"] == "relu":
        reinforcer_params["policy"]["feedforward"]["activations"] = torch.nn.ReLU()
    else:
        raise(KeyError)

    # Complete hidden dimsions
    reinforcer_params["policy"]["feedforward"]["hidden_dims"].append(len(select_augmenters))

    # Declare Optimizer
    if reinforcer_params["policy"]["optimizer"]["select_optimizer"] == "adam":
        reinforcer_params["policy"]["optimizer"]["select_optimizer"] = torch.optim.Adam
    else:
        raise(KeyError)

    reinforcer = REINFORCER(
        text_model.embedder,
        text_model.encoder,
        text_model.classifier,
        select_augmenters,
        vocab,
        reinforcer_params["environment"],
        reinforcer_params["policy"],
        reinforcer_params["REINFORCE"],
        dataset_dict
    )

    return reinforcer


def set_and_get_text_dataloader(
    dataloader_params: Dict,
    train_ds: AllennlpDataset,
    valid_ds: AllennlpDataset,
    test_ds: AllennlpDataset
):
    train_data_loader = DataLoader(
        train_ds,
        batch_size=dataloader_params["train_batch_size"],
        shuffle=dataloader_params["shuffle"],
        collate_fn=allennlp_collate
    )

    valid_data_loader = DataLoader(
        valid_ds,
        batch_size=dataloader_params["valid_batch_size"],
        collate_fn=allennlp_collate
    )

    test_data_loader = DataLoader(
        test_ds,
        batch_size=dataloader_params["test_batch_size"],
        collate_fn=allennlp_collate
    )

    return train_data_loader, valid_data_loader, test_data_loader


def set_and_get_reinforce_dataloader(
    dataloader_params: Dict,
    train_ds: AllennlpDataset
):
    reinforce_dataloader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=dataloader_params["shuffle"],
        collate_fn=allennlp_collate
    )

    return reinforce_dataloader


def set_and_get_synonyms(
    synonyms_params: Dict,
    train_ds: AllennlpDataset
):
    pass


def set_and_get_augmenters(
    augmenter_params: Dict,
    vocab: Vocabulary,
    embedder: Embedding,
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
        embedder,
        synonyms,
        augmenter_params["replace_augmenter"]
    )

    # Insert
    insert_augmenter = InsertAugmenter(
        vocab,
        embedder,
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


def set_and_get_text_trainer(
    text_trainer_params: Dict,
    sentiment_model: SentimentModel
):
    text_trainer = TextTrainer(
        text_trainer_params,
        sentiment_model
    )

    return text_trainer


def set_and_get_reinforce_trainer(
    reinforce_trainer_params: Dict,
    reinforcer: REINFORCER
):
    reinforce_trainer = ReinforceTrainer(
        reinforcer,
        is_writer=reinforce_trainer_params["is_writer"],
        is_save=reinforce_trainer_params["is_save"]
    )

    return reinforce_trainer


def set_and_save_augmented_texts(
    augmented_instances_params: Dict,
    dataset_reader: DatasetReader,
    train_ds: AllennlpDataset,
    reinforcer: REINFORCER
):
    if augmented_instances_params["select_policy"] != "none":
        if augmented_instances_params["num_processor"] == 1:
            generate_and_save_augmentation_texts(
                augmented_instances_params["select_policy"],
                augmented_instances_params["save_name"],
                dataset_reader,
                train_ds,
                reinforcer,
                augmented_instances_params["select_mode"]
            )
        elif augmented_instances_params["num_processor"] > 1:
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                raise RuntimeError

            pool = mp.Pool(augmented_instances_params["num_processor"])

            n = len(augmented_instances_params["select_policy"]) // augmented_instances_params["num_processor"]

            policy_args = [
                augmented_instances_params["select_policy"][i: i+n]
                for i in range(
                    0,
                    len(augmented_instances_params["select_policy"]),
                    n
                )
            ]

            save_name_args = [
                augmented_instances_params["save_name"][i: i+n]
                for i in range(
                    0,
                    len(augmented_instances_params["save_name"]),
                    n
                )
            ]

            with pool as p:
                p.starmap(
                    generate_and_save_augmentation_texts,
                    zip(
                        policy_args,
                        save_name_args,
                        repeat(dataset_reader),
                        repeat(train_ds),
                        repeat(reinforcer),
                        repeat(augmented_instances_params["select_mode"])
                    )
                )
        else:
            raise ValueError("Wrong number of the processor")
    else:
        pass


def set_augments_to_dataset_instances(
    dataset_dict: Dict,
    augmented_instances_save_names: List
):
    assert dataset_dict["dataset_reader"]._vocab is not None, "No vocab is given!"

    for save_name in augmented_instances_save_names:
        augment_texts = load_obj(save_name)
        print(augment_texts[400])

        for instance, augment_text in zip(dataset_dict["train_ds"].instances, augment_texts):
            field = TextField(
                dataset_dict["dataset_reader"]._tokenizer.tokenize(
                    augment_text
                ),
                dataset_dict["dataset_reader"]._indexers
            )
            instance.add_field(
                save_name,
                field,
                dataset_dict["dataset_reader"]._vocab
            )


def set_and_get_visualizer(
    visualizer_params: Dict,
    text_model: SentimentModel,
    vocab: Vocabulary
):
    if visualizer_params["mode"] == "tsne":
        visualizer = TSNEVisualizer(
            visualizer_params["tsne"],
            text_model.embedder,
            text_model.encoder,
            vocab
        )
    elif visualizer_params["mode"] == "isomap":
        visualizer = IsomapVisualizer(
            visualizer_params["isomap"],
            text_model.embedder,
            text_model.encoder,
            vocab
        )

    return visualizer
