import os
import tarfile
import pandas as pd

from typing import Dict
from pathlib import Path
from itertools import chain
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


DATA_PATH = "data/imdb.csv"


@DatasetReader.register("imdb")
class ImdbDatasetReader(DatasetReader):
    TAR_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    TRAIN_DIR = 'aclImdb/train'
    TEST_DIR = 'aclImdb/test'

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)

        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer}

    @overrides
    def _read(self, file_path):
        tar_path = cached_path(self.TAR_URL)
        tf = tarfile.open(tar_path, 'r')
        cache_dir = Path(os.path.dirname(tar_path))

        if not (cache_dir / self.TRAIN_DIR).exists() and not (cache_dir / self.TEST_DIR).exists():
            tf.extractall(cache_dir)
        else:
            pass

        if file_path == "train":
            pos_dir = os.path.join(self.TRAIN_DIR, "pos")
            neg_dir = os.path.join(self.TRAIN_DIR, "neg")
        elif file_path == "test":
            pos_dir = os.path.join(self.TEST_DIR, "pos")
            neg_dir = os.path.join(self.TEST_DIR, "neg")
        else:
            raise ValueError(f"only 'train' and 'test' are valid for 'file_path', but '{file_path}' is given.")

        path = chain(Path(cache_dir.joinpath(pos_dir)).glob("*.txt"),
                     Path(cache_dir.joinpath(neg_dir)).glob("*.txt"))

        for p in path:
            yield


def main():
    input_df = pd.read_csv(DATA_PATH)

    print(input_df)


if __name__ == '__main__':
    main()
