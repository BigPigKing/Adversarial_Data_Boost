from typing import Dict
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


class TextEmbedder(BasicTextFieldEmbedder):
    def __init__(self,
                 token_embedders: Dict):
        super(TextEmbedder, self).__init__(token_embedders)


def main():
    pass


if __name__ == '__main__':
    main()
