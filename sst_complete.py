import torch

from typing import Dict
from lib.configurer import get_config_params
from lib.configurer import set_and_get_dataset, set_and_get_vocab
from lib.configurer import set_and_get_text_model, set_and_get_reinforcer
from lib.configurer import set_and_get_text_trainer, set_and_get_reinforce_trainer
from lib.configurer import set_and_get_text_dataloader, set_and_get_reinforce_dataloader
from lib.configurer import set_and_save_augmented_sentences, get_augmented_instances


def train_text_model(
    mode_params: Dict,
    dataset_dict: Dict,
    text_model: torch.nn.Module
):
    # Get Text Trainer
    text_trainer = set_and_get_text_trainer(
        mode_params["text_trainer"],
        text_model
    )

    # Get Text DataLoader
    train_dataloader, valid_dataloader, test_dataloader = set_and_get_text_dataloader(
        mode_params["text_trainer"]["dataloader"],
        train_ds=dataset_dict["train_ds"],
        valid_ds=dataset_dict["valid_ds"],
        test_ds=dataset_dict["test_ds"]
    )

    # Train Text Model
    text_trainer.fit(
        mode_params["text_trainer"]["epochs"],
        train_dataloader,
        valid_dataloader,
        test_dataloader
    )


def train_reinforce_model(
    mode_params: Dict,
    dataset_dict: Dict,
    reinforcer: torch.nn.Module
):
    # Get Reinforce Trainer
    reinforce_trainer = set_and_get_reinforce_trainer(
        mode_params["reinforce_trainer"],
        reinforcer
    )

    # Get Reinforce Dataloader
    reinforce_dataloader = set_and_get_reinforce_dataloader(
        mode_params["reinforce_trainer"]["dataloader"],
        train_ds=dataset_dict["train_ds"]
    )

    # Train Reinforce Dataloader
    reinforce_trainer.fit(
        mode_params["reinforce_trainer"]["epochs"],
        mode_params["reinforce_trainer"]["batch_size"],
        reinforce_dataloader
    )


def generate_augmented_data(
    mode_params: Dict,
    dataset_dict: Dict,
    reinforcer: torch.nn.Module
):
    set_and_save_augmented_sentences(
        mode_params["augmented_instance_generator"],
        dataset_dict["dataset_reader"],
        dataset_dict["train_ds"],
        reinforcer
    )


def finetune_text_model(
    mode_params: Dict,
    dataset_dict: Dict,
    text_model: torch.nn.Module
):
    augmented_instances = get_augmented_instances(
        mode_params["text_finetuner"]["augmented_instance"]
    )

    dataset_dict["train_ds"].instances += augmented_instances

    # Get Text Trainer
    text_trainer = set_and_get_text_trainer(
        mode_params["text_finetuner"],
        text_model
    )

    # Get Text DataLoader
    train_dataloader, valid_dataloader, test_dataloader = set_and_get_text_dataloader(
        mode_params["text_finetuner"]["dataloader"],
        train_ds=dataset_dict["train_ds"],
        valid_ds=dataset_dict["valid_ds"],
        test_ds=dataset_dict["test_ds"]
    )

    # Train Text Model
    text_trainer.fit(
        mode_params["text_finetuner"]["epochs"],
        train_dataloader,
        valid_dataloader,
        test_dataloader
    )


def load_pretrained_text_model(
    mode_params: Dict,
    text_model: torch.nn.Module
):
    print("Loading pretrained weight for embedder ...")
    text_model.embedder.load_state_dict(
        torch.load(
            mode_params["pretrained_text_model"]["embedder"]
        )
    )

    print("Loading pretrained weight for encoder ...")
    text_model.encoder.load_state_dict(
        torch.load(
            mode_params["pretrained_text_model"]["encoder"]
        )
    )

    print("Loading pretrained weight for classifier ...")
    text_model.classifier.load_state_dict(
        torch.load(
            mode_params["pretrained_text_model"]["classifier"]
        )
    )


def all_procedure(
    mode_params: Dict,
    dataset_dict: Dict,
    text_model: torch.nn.Module,
    reinforcer: torch.nn.Module
):
    train_text_model(
        mode_params,
        dataset_dict,
        text_model
    )

    train_reinforce_model(
        mode_params,
        dataset_dict,
        reinforcer
    )

    generate_augmented_data(
        mode_params,
        dataset_dict,
        reinforcer
    )

    finetune_text_model(
        mode_params,
        dataset_dict,
        text_model
    )


def all_procedure_with_pretrained_text(
    mode_params: Dict,
    dataset_dict: Dict,
    text_model: torch.nn.Module,
    reinforcer: torch.nn.Module
):
    load_pretrained_text_model(
        mode_params,
        text_model
    )

    train_reinforce_model(
        mode_params,
        dataset_dict,
        reinforcer
    )

    generate_augmented_data(
        mode_params,
        dataset_dict,
        reinforcer
    )

    finetune_text_model(
        mode_params,
        dataset_dict,
        text_model
    )


def all_procedure_with_all_pretrained(
    mode_params: Dict,
    dataset_dict: Dict,
    text_model: torch.nn.Module,
    reinforcer: torch.nn.Module
):
    load_pretrained_text_model(
        mode_params,
        text_model
    )

    generate_augmented_data(
        mode_params,
        dataset_dict,
        reinforcer
    )

    finetune_text_model(
        mode_params,
        dataset_dict,
        text_model
    )


def finetune_procedure(
    mode_params: Dict,
    dataset_dict: Dict,
    text_model: torch.nn.Module,
    reinforcer: torch.nn.Module
):
    load_pretrained_text_model(
        mode_params,
        text_model
    )

    finetune_text_model(
        mode_params,
        dataset_dict,
        text_model
    )


def main(config_params):
    # Get Dataset
    dataset_dict = set_and_get_dataset(
        config_params["dataset"]
    )

    # Get Vocab
    vocab = set_and_get_vocab(
        dataset_dict
    )

    # Get Text Model
    text_model = set_and_get_text_model(
        config_params["text_model"],
        vocab
    )

    # Get Reinforcer
    reinforcer = set_and_get_reinforcer(
        config_params["reinforcer"],
        dataset_dict["train_ds"],
        vocab,
        text_model
    )

    # Move to GPU
    if config_params["env"]["USE_GPU"] is not None:
        text_model = text_model.cuda(config_params["env"]["USE_GPU"])
        reinforcer = reinforcer.cuda(config_params["env"]["USE_GPU"])
    else:
        pass

    # Go to Mode
    if config_params["train_mode"]["select_mode"] == 0:
        all_procedure(
            config_params["train_mode"]["0"],
            dataset_dict,
            text_model,
            reinforcer
        )
    elif config_params["train_mode"]["select_mode"] == 1:
        all_procedure_with_pretrained_text(
            config_params["train_mode"]["1"],
            dataset_dict,
            text_model,
            reinforcer
        )
    elif config_params["train_mode"]["select_mode"] == 2:
        all_procedure_with_all_pretrained(
            config_params["train_mode"]["2"],
            dataset_dict,
            text_model,
            reinforcer
        )
    elif config_params["train_mode"]["select_mode"] == 3:
        text_model.is_finetune = False
        finetune_procedure(
            config_params["train_mode"]["3"],
            dataset_dict,
            text_model,
            reinforcer
        )
    else:
        raise ValueError


if __name__ == '__main__':
    main(get_config_params("model_config.json"))
