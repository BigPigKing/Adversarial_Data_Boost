import pandas as pd
import torch.nn as nn
import torch.optim as optim

from typing import Dict, List
from overrides import overrides
from torch.utils.data import DataLoader, random_split
from allennlp.data import DatasetReader, Instance, allennlp_collate, AllennlpDataset
from allennlp.data.fields import LabelField, TextField, ListField
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.tokenizers.sentence_splitter import SentenceSplitter, SpacySentenceSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import GruSeq2SeqEncoder
from allennlp.modules.seq2vec_encoders import GruSeq2VecEncoder
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.modules import FeedForward
from allennlp.training import GradientDescentTrainer


DATA_PATH = "data/imdb.csv"


@DatasetReader.register("imdb")
class ImdbDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 sentence_splitter: SentenceSplitter = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)

        self._tokenizer = tokenizer or SpacyTokenizer()
        self._sentence_splitter = sentence_splitter or SpacySentenceSplitter()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        input_df = pd.read_csv(file_path)[:100]

        reviews = input_df["review"].to_list()
        sentiment_labels = input_df["sentiment"].to_list()

        # splited sentence
        splited_reviews = self._sentence_splitter.batch_split_sentences(reviews)

        # iterete over review
        for review_idx, splited_review in enumerate(splited_reviews):
            yield self.text_to_instance(splited_review, sentiment_labels[review_idx])

    @overrides
    def text_to_instance(self,
                         splited_review: List[List[str]],
                         sentiment_label: str):
        sentence_fields = []

        # iterate over sentence in review
        for sentence in splited_review:
            text_field = TextField(self._tokenizer.tokenize(sentence),
                                   token_indexers=self._token_indexers)
            sentence_fields.append(text_field)

        fields = {"review": ListField(sentence_fields), "label": LabelField(sentiment_label)}

        return Instance(fields)


def split_dataset(input_ds,
                  train_ratio=0.7,
                  valid_ratio=0.1):
    train_len = int(len(input_ds) * train_ratio)
    valid_len = int(len(input_ds) * valid_ratio)
    test_len = len(input_ds) - train_len - valid_len

    train_ds, valid_ds, test_ds = random_split(input_ds, [train_len, valid_len, test_len])

    return AllennlpDataset(train_ds), AllennlpDataset(valid_ds), AllennlpDataset(test_ds)


@Model.register("sentiment_classifier")
class Sentiment_Model(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: BasicTextFieldEmbedder,
        word_s2s_encoder: GruSeq2SeqEncoder,
        word_s2v_encoder: GruSeq2VecEncoder,
        sent_s2s_encoder: GruSeq2SeqEncoder,
        sent_s2v_encoder: GruSeq2VecEncoder,
        feedforward: FeedForward
    ):
        super().__init__(vocab)

        # Model Encoder Structure
        self.embedder = embedder
        self.word_s2s_encoder = word_s2s_encoder
        self.word_s2v_encoder = word_s2v_encoder
        self.sent_s2s_encoder = sent_s2s_encoder
        self.sent_s2v_encoder = sent_s2v_encoder

        # Model Output Structure
        self.feedforward = feedforward
        self.classification_layer = nn.Linear(self.feedforward.get_output_dim(), vocab.get_vocab_size("labels"))

    @overrides
    def forward(
        self,
        review,
        label
    ):
        print(review["tokens"]["tokens"].shape)
        review_mask = get_text_field_mask(review, num_wrapping_dims=1)
        print(review_mask.shape)
        print(label.shape)
        print("OK")


def main():
    # Set Dataset Reader
    imdb_dataset_reader = ImdbDatasetReader()
    imdb_ds = imdb_dataset_reader.read(DATA_PATH)

    # Split to train and test
    train_ds, valid_ds, test_ds = split_dataset(imdb_ds)

    # Set Vocabulary Set
    vocab = Vocabulary.from_instances(train_ds)
    train_ds.index_with(vocab)
    valid_ds.index_with(vocab)
    test_ds.index_with(vocab)

    # Batch begin
    train_data_loader = DataLoader(train_ds, batch_size=30, shuffle=True, collate_fn=allennlp_collate)
    valid_data_loader = DataLoader(valid_ds, batch_size=30, collate_fn=allennlp_collate)
    test_data_loader = DataLoader(test_ds, batch_size=30, collate_fn=allennlp_collate)

    # Embedder declartion
    glove_embedding = Embedding(
        embedding_dim=200,
        vocab=vocab,
        padding_index=0,
        pretrained_file="glove_200d.txt"
    )
    embedder = BasicTextFieldEmbedder(token_embedders={"tokens": glove_embedding})

    # Encoder declartion
    word_s2s_encoder = GruSeq2SeqEncoder(
        input_size=embedder.get_output_dim(),
        hidden_size=300,
        num_layers=2,
        bidirectional=True
    )
    word_s2v_encoder = GruSeq2VecEncoder(
        input_size=word_s2s_encoder.get_output_dim(),
        hidden_size=300,
        num_layers=1,
        bidirectional=True
    )
    sent_s2s_encoder = GruSeq2SeqEncoder(
        input_size=word_s2v_encoder.get_output_dim(),
        hidden_size=300,
        num_layers=2,
        bidirectional=True
    )
    sent_s2v_encoder = GruSeq2VecEncoder(
        input_size=sent_s2s_encoder.get_output_dim(),
        hidden_size=300,
        num_layers=1,
        bidirectional=True
    )

    # FeedForward declartion
    feedforward = FeedForward(
        input_dim=sent_s2v_encoder.get_output_dim(),
        num_layers=2,
        hidden_dims=[300, 150],
        activations=nn.ReLU(),
        dropout=0.3
    )

    # Model declartion
    gru_sentiment_model = Sentiment_Model(
        vocab,
        embedder,
        word_s2s_encoder,
        word_s2v_encoder,
        sent_s2s_encoder,
        sent_s2v_encoder,
        feedforward
    )

    # Trainer Declarition
    trainer = GradientDescentTrainer(
        gru_sentiment_model,
        optim.RMSprop(gru_sentiment_model.parameters()),
        train_data_loader
    )

    # Trainer Train
    trainer.train()


if __name__ == '__main__':
    main()
