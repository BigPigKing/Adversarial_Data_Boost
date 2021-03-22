import torch
import logging

from lib.dataset import get_sst_ds
from lib.embedder import TextEmbedder
from lib.encoder import TextEncoder
from lib.classifier import TextClassifier
from lib.model import SentimentModel
from lib.trainer import TextTrainer, ReinforceTrainer
from lib.augmenter import DeleteAugmenter, SwapAugmenter, IdentityAugmenter, InsertAugmenter, ReplaceAugmenter
from lib.reinforcer import REINFORCER
from lib.utils import add_wordnet_to_vocab, augment_and_add_instances_to_dataset, get_synonyms_from_dataset, save_obj, load_obj

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from allennlp.data import allennlp_collate
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding


logger = logging.getLogger(__name__)
USE_GPU = torch.cuda.is_available()
IS_LOAD_SYNONYM_DICT = True
IS_PRETRAINED_TEXT = True
IS_PRETRAINED_REINFORCE = True


def main():
    # Get Dataset
    train_ds, valid_ds, test_ds, dataset_reader = get_sst_ds(granularity="5-class")

    # Set Vocabulary Set
    vocab = Vocabulary.from_instances(train_ds)
    vocab = add_wordnet_to_vocab(vocab)

    train_ds.index_with(vocab)
    valid_ds.index_with(vocab)
    test_ds.index_with(vocab)

    # Batch begin
    train_data_loader = DataLoader(train_ds, batch_size=200, shuffle=True, collate_fn=allennlp_collate)
    valid_data_loader = DataLoader(valid_ds, batch_size=200, collate_fn=allennlp_collate)
    test_data_loader = DataLoader(test_ds, batch_size=200, collate_fn=allennlp_collate)

    # Embedder declartion
    glove_embedding = Embedding(
        embedding_dim=300,
        vocab=vocab,
        padding_index=0,
        pretrained_file="pretrained_weight/glove.6B.300d.txt"
    )
    embedder = TextEmbedder(pretrained_embedding=glove_embedding)

    # Encoder declartion
    encoder = TextEncoder(input_size=embedder.get_output_dim())

    # Classifier declartion
    classifier = TextClassifier(
        input_size=encoder.get_output_dim(),
        output_size=vocab.get_vocab_size("labels")
    )

    # Schedular declartion
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

    # Get synonyms dictionary
    if IS_LOAD_SYNONYM_DICT is False:
        synonym_dict = get_synonyms_from_dataset(train_ds)
        save_obj(synonym_dict, "sst_synonyms")
    else:
        synonym_dict = load_obj("sst_synonyms")

    # Augmenter declartion
    delete_augmenter = DeleteAugmenter()
    swap_augmenter = SwapAugmenter()
    insert_augmenter = InsertAugmenter(vocab=vocab, synonym_dict=synonym_dict)
    replace_augmenter = ReplaceAugmenter(vocab=vocab, synonym_dict=synonym_dict)
    identity_augmenter = IdentityAugmenter()

    # Reinforcer declation
    reinforcer = REINFORCER(
        embedder,
        encoder,
        classifier,
        [delete_augmenter, swap_augmenter, insert_augmenter, replace_augmenter, identity_augmenter],
        vocab=vocab
    )

    # Model declartion
    sentiment_model = SentimentModel(
        vocab,
        embedder,
        encoder,
        classifier,
        reinforcer,
    )

    # Writer declartion
    writer = SummaryWriter()

    # Model move to gpu
    if USE_GPU is True:
        sentiment_model = sentiment_model.cuda()

    # Text trainer declartion
    if IS_PRETRAINED_TEXT is True:
        sentiment_model.encoder.load_state_dict(torch.load("encoder.pkl"))
        sentiment_model.classifier.load_state_dict(torch.load("classifier.pkl"))
    else:
        text_trainer = TextTrainer(sentiment_model)
        text_trainer.fit(24, train_data_loader, valid_data_loader, test_data_loader)
        torch.save(sentiment_model.encoder.state_dict(), "encoder.pkl")
        torch.save(sentiment_model.classifier.state_dict(), "classifier.pkl")

    # REINFORCE trainer declartion
    if IS_PRETRAINED_REINFORCE is True:
        sentiment_model.reinforcer.policy.load_state_dict(torch.load("policy23.pkl"))
    else:
        train_data_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=allennlp_collate)
        reinforce_trainer = ReinforceTrainer(reinforcer, writer)
        reinforce_trainer.fit(2, 600, train_data_loader, is_save=True)

    augment_instances = []
    augment_instances += load_obj("augment")

    sentiment_model.reinforcer.policy.load_state_dict(torch.load("policy23.pkl"))

    augment_instances += augment_and_add_instances_to_dataset(
        dataset_reader,
        train_ds,
        reinforcer
    )

    sentiment_model.reinforcer.policy.load_state_dict(torch.load("policy21.pkl"))

    augment_instances += augment_and_add_instances_to_dataset(
        dataset_reader,
        train_ds,
        reinforcer
    )

    train_ds.instances += augment_instances

    train_data_loader = DataLoader(train_ds, batch_size=200, shuffle=True, collate_fn=allennlp_collate)
    text_trainer.fit(30, train_data_loader, valid_data_loader, test_data_loader)


if __name__ == '__main__':
    main()
