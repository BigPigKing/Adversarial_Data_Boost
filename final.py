import pandas as pd


DATA_PATH = "data/imdb.csv"


def main():
    input_df = pd.read_csv(DATA_PATH)

    print(input_df)


if __name__ == '__main__':
    main()
