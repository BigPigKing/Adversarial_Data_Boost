import csv
import logging
import numpy as np

from .tokenizer import WordTokenizer

from typing import Dict, Optional, Union
from overrides import overrides
from nltk.tree import Tree

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.fields import LabelField, TextField, Field

logger = logging.getLogger(__name__)


@DatasetReader.register("yelp_tokens")
class YelpReviewDatasetReader(DatasetReader):
    def __init__(
        self,
        yelp_params: Dict,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if yelp_params["transformer_model_name"] is None:
            self._indexers = {
                "tokens": SingleIdTokenIndexer()
            }
            self._tokenizer = WordTokenizer()
            self._vocab = None
            self.is_transformer = False
        else:
            self._indexers = {
                "tokens": PretrainedTransformerIndexer(
                    yelp_params["transformer_model_name"]
                )
            }
            self._tokenizer = PretrainedTransformerTokenizer(
                yelp_params["transformer_model_name"]
            )
            self._vocab = Vocabulary.from_pretrained_transformer(
                yelp_params["transformer_model_name"]
            )
            self.is_transformer = True
        self.detokenizer = WordTokenizer()

        self.field_names = {
            "text": [yelp_params["review_field_name"]],
            "label": [yelp_params["label_field_name"]],
            "augments": []
        }  # Warning: Augments now only suitable for one original field

    @overrides
    def _read(self, file_path):
        with open(file_path, "r") as data_file:
            reader = csv.reader(data_file, delimiter=",")

            for i, line in enumerate(reader):
                review = line[1]
                label = line[0]
                instance = self.text_to_instance(review, label)

                if instance is not None:
                    yield instance
                else:
                    pass

    def make_token(
        self,
        t: Union[str, Token]
    ):
        if isinstance(t, str):
            return Token(t)
        elif isinstance(t, Token):
            return t
        else:
            raise ValueError("Tokens must be either str or Token.")

    @overrides
    def text_to_instance(
        self,
        text: str,
        sentiment: str = None
    ) -> Optional[Instance]:
        tokens = self._tokenizer.tokenize(text)

        if self.is_transformer is False:
            tokens = [self.make_token(x) for x in tokens]
        else:
            pass

        text_field = TextField(
            tokens,
            token_indexers=self._indexers
        )
        fields: Dict[str, Field] = {
            self.field_names["text"][0]: text_field
        }

        if sentiment is not None:
            fields[self.field_names["label"][0]] = LabelField(sentiment)
        else:
            pass

        return Instance(fields)

    def get_token_indexers(self):
        return self._token_indexers


@DatasetReader.register("sst_tokens")
class StanfordSentimentTreeBankDatasetReader(DatasetReader):
    """
    Reads tokens and their sentiment labels from the Stanford Sentiment Treebank.
    The Stanford Sentiment Treebank comes with labels
    from 0 to 4. `"5-class"` uses these labels as is. `"3-class"` converts the
    problem into one of identifying whether a sentence is negative, positive, or
    neutral sentiment. In this case, 0 and 1 are grouped as label 0 (negative sentiment),
    2 is converted to label 1 (neutral sentiment) and 3 and 4 are grouped as label 2
    (positive sentiment). `"2-class"` turns it into a binary classification problem
    between positive and negative sentiment. 0 and 1 are grouped as the label 0
    (negative sentiment), 2 (neutral) is discarded, and 3 and 4 are grouped as the label 1
    (positive sentiment).
    Expected format for each input line: a linearized tree, where nodes are labeled
    by their sentiment.
    The output of `read` is a list of `Instance` s with the fields:
        tokens : `TextField` and
        label : `LabelField`
    Registered as a `DatasetReader` with name "sst_tokens".
    # Parameters
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    use_subtrees : `bool`, optional, (default = `False`)
        Whether or not to use sentiment-tagged subtrees.
    granularity : `str`, optional (default = `"5-class"`)
        One of `"5-class"`, `"3-class"`, or `"2-class"`, indicating the number
        of sentiment labels to use.
    """

    def __init__(
        self,
        sst_params: Dict,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if sst_params["transformer_model_name"] is None:
            self._indexers = {
                "tokens": SingleIdTokenIndexer()
            }
            self._tokenizer = WordTokenizer()
            self._vocab = None
            self.is_transformer = False
        else:
            self._indexers = {
                "tokens": PretrainedTransformerIndexer(
                    sst_params["transformer_model_name"]
                )
            }
            self._tokenizer = PretrainedTransformerTokenizer(
                sst_params["transformer_model_name"]
            )
            self._vocab = Vocabulary.from_pretrained_transformer(
                sst_params["transformer_model_name"]
            )
            self.is_transformer = True
        self._use_subtrees = sst_params["use_subtrees"]
        self.detokenizer = WordTokenizer()
        self.max_length = sst_params["max_length"]

        allowed_granularities = ["5-class", "3-class", "2-class"]

        if sst_params["granularity"] not in allowed_granularities:
            raise ConfigurationError(
                "granularity is {}, but expected one of: {}".format(
                    sst_params["granularity"], allowed_granularities
                )
            )
        self._granularity = sst_params["granularity"]
        self.field_names = {
            "text": [sst_params["review_field_name"]],
            "label": [sst_params["label_field_name"]],
            "augments": []
        }  # Warning: Augments now only suitable for one original field

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file.readlines():
                line = line.strip("\n")
                if not line:
                    continue
                parsed_line = Tree.fromstring(line)
                if self._use_subtrees:
                    for subtree in parsed_line.subtrees():
                        instance = self.text_to_instance(
                            self.detokenizer.detokenize(subtree.leaves()),
                            subtree.label()
                        )
                        if instance is not None:
                            yield instance
                else:
                    instance = self.text_to_instance(
                        self.detokenizer.detokenize(parsed_line.leaves()),
                        parsed_line.label()
                    )
                    if instance is not None:
                        yield instance

    def make_token(
        self,
        t: Union[str, Token]
    ):
        if isinstance(t, str):
            return Token(t)
        elif isinstance(t, Token):
            return t
        else:
            raise ValueError("Tokens must be either str or Token.")

    @overrides
    def text_to_instance(
        self,
        text: str,
        sentiment: str = None
    ) -> Optional[Instance]:
        tokens = self._tokenizer.tokenize(text)

        if self.is_transformer is False:
            tokens = [self.make_token(x) for x in tokens]
        else:
            pass

        text_field = TextField(
            tokens[:self.max_length],
            token_indexers=self._indexers
        )
        fields: Dict[str, Field] = {
            self.field_names["text"][0]: text_field
        }

        if sentiment is not None:
            if self._granularity == "3-class":
                if int(sentiment) < 2:
                    sentiment = "0"
                elif int(sentiment) == 2:
                    sentiment = "1"
                else:
                    sentiment = "2"
            elif self._granularity == "2-class":
                if int(sentiment) < 2:
                    sentiment = "0"
                elif int(sentiment) == 2:
                    return None
                else:
                    sentiment = "1"
            fields[self.field_names["label"][0]] = LabelField(sentiment)
        else:
            pass

        return Instance(fields)

    def get_token_indexers(self):
        return self._token_indexers


def get_sst_ds(
    sst_params: Dict,
    train_data_path="data/sst/train.txt",
    valid_data_path="data/sst/dev.txt",
    test_data_path="data/sst/test.txt",
):
    sst_dataset_reader = StanfordSentimentTreeBankDatasetReader(
        sst_params
    )

    if sst_params["proportion"] != 1:
        train_ds = sst_dataset_reader.read(train_data_path + str(sst_params["proportion"]))
    else:
        train_ds = sst_dataset_reader.read(train_data_path)
    valid_ds = sst_dataset_reader.read(valid_data_path)
    test_ds = sst_dataset_reader.read(test_data_path)

    return train_ds, valid_ds, test_ds, sst_dataset_reader


def get_yelp_ds(
    train_data_path="data/yelp_review_full/train.csv",
    valid_data_path="data/yelp_review_full/valid.csv",
    test_data_path="data/yelp_review_full/test.csv",
    train_data_proportion=1
):
    yelp_dataset_reader = YelpReviewDatasetReader()
    train_ds = yelp_dataset_reader.read(train_data_path)
    valid_ds = yelp_dataset_reader.read(valid_data_path)
    test_ds = yelp_dataset_reader.read(test_data_path)

    return train_ds, valid_ds, test_ds, yelp_dataset_reader


def split_proportion_csv(
    file_path,
    output_path,
    proportion,
    delimiter=","
):
    from sklearn.model_selection import train_test_split

    with open(file_path, 'r') as f:
        lines = f.read().splitlines()

    train, test = train_test_split(lines, train_size=proportion)

    with open(file_path + str(proportion), 'w+') as f:
        for line in train:
            f.write(line+delimiter)


def main():
    sst_path = "../data/sst/train.txt"
    sst_dir_path = "../data/sst/"

    for proportion in np.arange(0.1, 1, 0.1):
        split_proportion_csv(
            sst_path,
            sst_dir_path,
            proportion,
            delimiter='\n'
        )


if __name__ == '__main__':
    main()
