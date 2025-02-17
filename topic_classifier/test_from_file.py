import json
import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import DATA_NAME


def create_dataset() -> None:
    """
    Creates dataset from the csv file
    """
    data_dir = os.path.join("data")
    data_path = os.path.join(data_dir, DATA_NAME)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Дирректория '{data_dir}' не найдена.")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Файл '{data_path}' не найден.")

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Произошла ошибка при чтении файла: {e}")

    dataset = set()
    for i in range(len(df)):
        questions = df["questions"][i].split(",")
        category = df["category"][i]
        for question in questions:
            question = question.strip("}{").strip()
            dataset.add((question, category))

    with open(os.path.join("data", "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(list(dataset), f, ensure_ascii=False, indent=4)

    logging.info("Data: " + str(len(df)))
    logging.info("Final dataset: " + str(len(dataset)))


def load_data(test_size=0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads dataset from json file and returns train and test datasets
    :param test_size: size of test dataset
    :return: train and test datasets respectively
    """
    with open(os.path.join("data", "dataset.json"), "r", encoding="utf-8") as f:
        dataset = json.load(f)
    columns = ["question", "category"]
    df_dataset = pd.DataFrame(dataset, columns=columns)
    print(df_dataset.head())
    train_df, test_df = train_test_split(
        df_dataset, test_size=test_size, random_state=42
    )
    return train_df, test_df
