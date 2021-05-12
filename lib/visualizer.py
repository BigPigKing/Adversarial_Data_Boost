import abc
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .encoder import TextEncoder
from .embedder import TextEmbedder
from tqdm import tqdm
from typing import Dict
from overrides import overrides
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from allennlp.data import allennlp_collate
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import move_to_device


class visualizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _reduce_dim(
        input_X: torch.Tensor
    ):
        return NotImplemented

    @abc.abstractmethod
    def _encode(
        batch: Dict
    ):
        return NotImplemented

    @abc.abstractmethod
    def visualize(
        self,
        ds: AllennlpDataset
    ):
        return NotImplemented


class TSNEVisualizer(visualizer):
    def __init__(
        self,
        visualizer_params: Dict,
        embedder: TextEmbedder,
        encoder: TextEncoder,
        vocab: Vocabulary,
    ):
        self.embedder = embedder
        self.encoder = encoder
        self.vocab = vocab
        self.col_names = visualizer_params["col_names"]
        self.reducer = TSNE(
            n_components=visualizer_params["n_components"],
            perplexity=visualizer_params["perplexity"],
            n_iter=visualizer_params["n_iter"],
            verbose=visualizer_params["verbose"]
        )
        self.save_fig_path = visualizer_params["save_fig_path"] + "fig"

    @overrides
    def _encode(
        self,
        token_X
    ):
        with torch.no_grad():
            embed_X = self.embedder(token_X)
            tokens_mask = get_text_field_mask(token_X)
            encode_X = self.encoder(embed_X.detach(), tokens_mask)

        return encode_X

    @overrides
    def _reduce_dim(
        self,
        encode_X: torch.Tensor
    ):
        return self.reducer.fit_transform(
            encode_X
        )

    def _get_plot_df(
        self,
        total_X: np.ndarray,
        total_Y: np.ndarray,
        total_Z: np.ndarray
    ):
        # total_Z responsible for indexing, 1 means the original data
        total_Z = total_Z == 1

        # First the orignal one
        origin_X = total_X[total_Z]
        origin_Y = total_Y[total_Z]
        origin_Y = np.array(
            list(
                map(
                    lambda x: self.vocab.get_token_from_index(
                        x,
                        namespace="labels"
                    ),
                    list(origin_Y)
                )
            )
        )

        # Then the augment one
        augment_X = total_X[~total_Z]
        augment_Y = total_Y[~total_Z]
        augment_Y = np.array(
            list(
                map(
                    lambda x: self.vocab.get_token_from_index(
                        x,
                        namespace="labels"
                    ) + "_aug",
                    list(augment_Y)
                )
            )
        )

        # Union
        union_X = np.vstack([origin_X, augment_X])
        del origin_X, augment_X
        union_Y = np.concatenate([origin_Y, augment_Y])
        union_Y = union_Y.reshape([union_Y.shape[0], 1])
        del origin_Y, augment_Y
        union_arr = np.hstack([union_X, union_Y])
        del union_X, union_Y

        # Get Df
        plot_df = pd.DataFrame(
            union_arr,
            columns=self.col_names
        ).astype({
            self.col_names[0]: "float32",
            self.col_names[1]: "float32",
            self.col_names[2]: "category"
        })
        del union_arr

        return plot_df

    def _visualize(
        self,
        total_X: np.ndarray,
        total_Y: np.ndarray,
        total_Z: np.ndarray
    ):
        plot_df = self._get_plot_df(
            total_X,
            total_Y,
            total_Z
        )

        plot_df = plot_df.sample(frac=0.1)

        sns.scatterplot(
            data=plot_df,
            x=self.col_names[0],
            y=self.col_names[1],
            hue=self.col_names[2],
            hue_order=list(plot_df[self.col_names[2]].unique()).sort(),
            style=self.col_names[2],
            style_order=list(plot_df[self.col_names[2]].unique()).sort(),
            legend="full",
            palette=sns.color_palette(
                "hls",
                plot_df[self.col_names[2]].nunique()
            ),
            alpha=1,
            s=15
        )

        plt.savefig(
            self.save_fig_path,
            dpi=1200
        )

    @overrides
    def visualize(
        self,
        ds: AllennlpDataset,
        batch_size: int = 1200
    ):
        dataloader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=allennlp_collate
        )

        total_X = []
        total_Y = []
        total_Z = []

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            batch = move_to_device(batch, 0)

            token_X = batch["tokens"]
            label_Y = batch["label"].detach().cpu().numpy()
            augment_Z = np.array(batch["augment"])

            encode_X = self._encode(token_X).cpu().numpy()

            total_X.append(encode_X)
            total_Y.append(label_Y)
            total_Z.append(augment_Z)

        total_X = np.vstack(total_X)
        total_X = self._reduce_dim(total_X)
        total_Y = np.concatenate(total_Y)
        total_Z = np.concatenate(total_Z)

        self._visualize(
            total_X,
            total_Y,
            total_Z
        )


def main():
    pass


if __name__ == '__main__':
    main()
