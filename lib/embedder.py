from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


class TextEmbedder(BasicTextFieldEmbedder):
    def __init__(self,
                 pretrained_embedding: Embedding = None):
        super(TextEmbedder, self).__init__(token_embedders={"tokens": pretrained_embedding})


def main():
    pass


if __name__ == '__main__':
    main()
