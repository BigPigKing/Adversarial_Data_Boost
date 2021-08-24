import textattack
import transformers
import pandas as pd


def get_dataset(
    datapath: str
):
    input_df = pd.read_csv(datapath)
    sentences = input_df["sentence"].to_list()
    labels = input_df["label"].to_list()
    pre_dataset = list(zip(sentences, labels))

    return textattack.datasets.Dataset(pre_dataset)


def main():
    train_ds = get_dataset("data/sst/sst_2/SST_train.csv")
    test_ds = get_dataset("data/sst/sst_2/SST_test.csv")
    model = transformers.AutoModelForSequenceClassification.from_pretrained("roberta-base")
    tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper)
    training_args = textattack.TrainingArgs(
        num_epochs=15,
        num_clean_epochs=3,
        attack_epoch_interval=3,
        num_train_adv_examples=6000,
        learning_rate=2e-5,
        num_warmup_steps=0.06,
        attack_num_workers_per_device=9,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        log_to_tb=True,
    )
    trainer = textattack.Trainer(
        model_wrapper,
        "classification",
        attack,
        train_ds,
        test_ds,
        training_args
    )

    trainer.train()

    noisy_ds = []
    noisy_ds.append(get_dataset("data/sst/sst_2/SST_stack_eda.csv"))
    noisy_ds.append(get_dataset("data/sst/sst_2/SST_eda.csv"))
    noisy_ds.append(get_dataset("data/sst/sst_2/SST_embedding.csv"))
    noisy_ds.append(get_dataset("data/sst/sst_2/SST_clare.csv"))
    noisy_ds.append(get_dataset("data/sst/sst_2/SST_checklist.csv"))
    noisy_ds.append(get_dataset("data/sst/sst_2/SST_char.csv"))
    noisy_ds.append(get_dataset("data/sst/sst_2/SST_backtrans_de.csv"))
    noisy_ds.append(get_dataset("data/sst/sst_2/SST_backtrans_ru.csv"))
    noisy_ds.append(get_dataset("data/sst/sst_2/SST_backtrans_zh.csv"))
    noisy_ds.append(get_dataset("data/sst/sst_2/SST_spell.csv"))

    for nds in noisy_ds:
        trainer = textattack.Trainer(
            model_wrapper,
            "classification",
            attack,
            train_ds,
            nds,
            training_args
        )
        trainer.evaluate()


if __name__ == '__main__':
    main()
