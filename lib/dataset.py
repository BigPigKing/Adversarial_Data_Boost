import csv
import logging
import numpy as np

from typing import Dict, List, Optional, Union
from overrides import overrides
from nltk.tree import Tree

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import LabelField, TextField, Field, MetadataField

logger = logging.getLogger(__name__)


class YelpReviewDatasetReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Optional[Tokenizer] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokenizer = tokenizer or SpacyTokenizer(split_on_spaces=True)

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

    @overrides
    def text_to_instance(self, review: str, sentiment: str = None) -> Optional[Instance]:
        """
        We take `pre-tokenized` input here, because we might not have a tokenizer in this class.
        # Parameters
        tokens : `List[str]`, required.
            The tokens in a given sentence.
        sentiment : `str`, optional, (default = `None`).
            The sentiment for this sentence.
        # Returns
        An `Instance` containing the following fields:
            tokens : `TextField`
                The tokens in the sentence or phrase.
            label : `LabelField`
                The sentiment label of the sentence or phrase.
        """
        assert isinstance(
            review, str
        )

        tokens = self._tokenizer.tokenize(review)

        text_field = TextField(tokens, token_indexers=self._token_indexers)
        label_field = LabelField(sentiment)

        fields: Dict[str, Field] = {
            "tokens": text_field,
            "label": label_field
        }

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
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Optional[Tokenizer] = None,
        use_subtrees: bool = False,
        granularity: str = "5-class",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokenizer = tokenizer or SpacyTokenizer(split_on_spaces=True)
        self._use_subtrees = use_subtrees
        allowed_granularities = ["5-class", "3-class", "2-class"]
        if granularity not in allowed_granularities:
            raise ConfigurationError(
                "granularity is {}, but expected one of: {}".format(
                    granularity, allowed_granularities
                )
            )
        self._granularity = granularity

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
                        instance = self.text_to_instance(subtree.leaves(), subtree.label())
                        if instance is not None:
                            yield instance
                else:
                    instance = self.text_to_instance(parsed_line.leaves(), parsed_line.label())
                    if instance is not None:
                        yield instance

    @overrides
    def text_to_instance(
        self,
        tokens: List[str],
        sentiment: str = None,
        augment: int = 1
    ) -> Optional[Instance]:
        """
        We take `pre-tokenized` input here, because we might not have a tokenizer in this class.
        # Parameters
        tokens : `List[str]`, required.
            The tokens in a given sentence.
        sentiment : `str`, optional, (default = `None`).
            The sentiment for this sentence.
        # Returns
        An `Instance` containing the following fields:
            tokens : `TextField`
                The tokens in the sentence or phrase.
            label : `LabelField`
                The sentiment label of the sentence or phrase.
        """
        assert isinstance(
            tokens, list
        )  # If tokens is a str, nothing breaks but the results are garbage, so we check.
        if self._tokenizer is None:

            def make_token(t: Union[str, Token]):
                if isinstance(t, str):
                    return Token(t)
                elif isinstance(t, Token):
                    return t
                else:
                    raise ValueError("Tokens must be either str or Token.")

            tokens = [make_token(x) for x in tokens]
        else:
            tokens = self._tokenizer.tokenize(" ".join(tokens))
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}
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
            fields["label"] = LabelField(sentiment)

        augment_field = MetadataField(
            augment
        )

        fields["augment"] = augment_field

        return Instance(fields)

    def get_token_indexers(self):
        return self._token_indexers


def get_sst_ds(
    train_data_path="data/sst/train.txt",
    valid_data_path="data/sst/dev.txt",
    test_data_path="data/sst/test.txt",
    train_data_proportion=1,
    granularity="2-class"
):
    sst_dataset_reader = StanfordSentimentTreeBankDatasetReader(granularity=granularity)

    if train_data_proportion != 1:
        train_ds = sst_dataset_reader.read(train_data_path + str(train_data_proportion))
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
