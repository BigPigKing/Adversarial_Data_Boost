import torch

from typing import Dict
from allennlp.modules.seq2seq_encoders import GruSeq2SeqEncoder
from allennlp.modules.seq2vec_encoders import GruSeq2VecEncoder


class TextEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        s2s_encoder_params: Dict,
        s2v_encoder_params: Dict
    ):
        super(TextEncoder, self).__init__()

        s2s_encoder = GruSeq2SeqEncoder(
            input_size=input_size,
            hidden_size=s2s_encoder_params["hidden_size"],
            num_layers=s2s_encoder_params["num_layers"],
            bidirectional=s2s_encoder_params["bidirectional"]
        )

        layer_norm_1 = torch.nn.LayerNorm(s2s_encoder.get_output_dim())

        s2v_encoder = GruSeq2VecEncoder(
            input_size=s2s_encoder.get_output_dim(),
            hidden_size=s2v_encoder_params["hidden_size"],
            num_layers=s2v_encoder_params["num_layers"],
            bidirectional=s2v_encoder_params["bidirectional"]
        )

        layer_norm_2 = torch.nn.LayerNorm(s2v_encoder.get_output_dim())

        self.encoders = torch.nn.ModuleList([s2s_encoder, layer_norm_1, s2v_encoder, layer_norm_2])

    def forward(
        self,
        embed_X,
        tokens_mask
    ):
        # Iterate over encoder to produce encoding X
        for encoder in self.encoders:
            if type(encoder) == torch.nn.LayerNorm:
                embed_X = encoder(embed_X)
            else:
                embed_X = encoder(embed_X, tokens_mask)

        return embed_X

    def get_output_dim(self) -> int:
        return self.encoders[-2].get_output_dim()


def main():
    pass


if __name__ == '__main__':
    main()
