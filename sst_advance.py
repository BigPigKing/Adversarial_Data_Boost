import torch
import logging

from lib.dataset import get_sst_ds
from lib.embedder import TextEmbedder
from lib.encoder import TextEncoder
from lib.classifier import TextClassifier
from lib.model import SentimentModel
from lib.trainer import TextTrainer
from lib.augmenter import DeleteAugmenter
from lib.reinforcer import REINFORCER

from torch.utils.data import DataLoader
from allennlp.data import allennlp_collate
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding


logger = logging.getLogger(__name__)
USE_GPU = torch.cuda.is_available()


def main():
    # Get Dataset
    train_ds, valid_ds, test_ds = get_sst_ds(granularity="5-class")

    # Set Vocabulary Set
    vocab = Vocabulary.from_instances(train_ds)
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

    # Augmenter declartion
    augmenter = DeleteAugmenter(0)

    # Reinforcer declation
    reinforcer = REINFORCER(
        embedder,
        encoder,
        [augmenter]
    )

    # Model declartion
    sentiment_model = SentimentModel(
        vocab,
        embedder,
        encoder,
        classifier,
        reinforcer,
    )

    # Model move to gpu
    if USE_GPU is True:
        sentiment_model = sentiment_model.cuda()

    # Trainer declartion
    trainer = TextTrainer(sentiment_model)

    trainer.fit(500, train_data_loader, valid_data_loader, test_data_loader)


if __name__ == '__main__':
    main()
