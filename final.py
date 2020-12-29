import pandas as pd

from typing import Dict
from overrides import overrides
from torch.utils.data import DataLoader
from allennlp.data import DatasetReader, Instance, allennlp_collate
from allennlp.data.fields import LabelField, TextField, ListField
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.tokenizers.sentence_splitter import SentenceSplitter, SpacySentenceSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary


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
            sentence_fields = []

            # iterate over sentence in review
            for sentence in splited_review:
                text_field = TextField(self._tokenizer.tokenize(sentence),
                                       token_indexers=self._token_indexers)
                sentence_fields.append(text_field)

            fields = {"review": ListField(sentence_fields), "label": LabelField(sentiment_labels[review_idx])}

            yield Instance(fields)


def main():
    # Set Dataset Reader
    imdb_dataset_reader = ImdbDatasetReader()
    imdb_ds = imdb_dataset_reader.read(DATA_PATH)

    # Set Vocabulary Set
    vocab = Vocabulary.from_instances(imdb_ds)
    imdb_ds.index_with(vocab)

    # Batch begin
    data_loader = DataLoader(imdb_ds, batch_size=3, collate_fn=allennlp_collate)

    for batch in data_loader:
        print(batch["review"]["tokens"]["tokens"].shape)


if __name__ == '__main__':
    main()
