import json
import logging
import os
import time
from typing import Dict, List, Optional

import onnx
import openai
import pandas as pd
import requests
import torch
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from tenacity import retry, stop_after_attempt, wait_exponential
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import wandb

from .private_api import WANDB_API
from .utils import CHECK_DND_RELATION, LMSTUDIO_URL, MODEL_NAME

client = openai.OpenAI(base_url=LMSTUDIO_URL, api_key="<KEY>")


class OneCycleTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=self.args.learning_rate,  # peak learning rate
            total_steps=num_training_steps,
            pct_start=0.1,  # percentage of steps increasing the LR
            anneal_strategy="linear",  # or "cos"
            div_factor=25.0,  # initial_lr = max_lr / div_factor
        )
        return scheduler


def tokenize_function(
    example: dict, tokenizer: torch.nn.Module | AutoTokenizer, max_length: int
) -> Dict:
    """
    Tokenize text for classification.
    """
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def final_test_model(
    model: torch.nn.Module,
    json_path: str,
    tokenizer,  # AutoTokenizer or similar
    max_length: int,
    batch_size: int = 16,
):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = [{"text": item[0], "label": item[1]} for item in raw_data]
    dataset = Dataset.from_list(data)

    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer=tokenizer, max_length=max_length),
        batched=True,
    )
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    dataloader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    all_predictions = []
    all_labels = []

    for batch in dataloader:
        inputs = {
            k: v.to(model.device)
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask"]
        }
        labels = batch["labels"].to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")

    return {"accuracy": accuracy, "f1": f1}


def test_llm_classifier(json_path: str):
    """
    Loads the dataset from a JSON file,
    sends each text sample to the LLM classifier endpoint using the payload
    {"messages": "<text>"},  and calculates
    accuracy and weighted F1 score.

    The dataset is expected to be a JSON array of lists, e.g.:
      [
         ["Sample text 1", 1],
         ["Sample text 2", 0],
         ...
      ]

    The endpoint should accept a JSON payload: {"messages": "<text>"}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data = [{"text": item[0], "label": item[1]} for item in raw_data]

    all_predictions = []
    all_labels = []

    for sample in data:
        text = sample["text"]
        true_label = sample["label"]
        payload = {"messages": text}
        try:
            response = requests.post(CHECK_DND_RELATION, json=payload)
            response.raise_for_status()
            result = response.json()
            pred_text = result.get("related_to_dnd")
            if pred_text is None:
                logging.error(f"No prediction found in response for text: {text}")
                continue
            if pred_text:
                pred = 1
            elif not pred_text:
                pred = 0
            else:
                logging.error(
                    f"Unexpected prediction value '{pred_text}' for text: {text}"
                )
                continue

            all_predictions.append(pred)
            all_labels.append(true_label)
        except Exception as e:
            logging.error(f"Error processing sample '{text}': {e}")

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    logging.info(f"LLM Classifier Accuracy: {accuracy:.4f}")
    logging.info(f"LLM Classifier F1: {f1:.4f}")

    return {"accuracy": accuracy, "f1": f1}


@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_translate(prompt: str) -> str:
    """
    Call OpenAI ChatCompletion API to translate a given prompt.

    Args:
        prompt (str): The text prompt containing content to translate.

    Returns:
        str: The translated text returned by the API.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful translation assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def batch_translate(
    texts: List[str], target_lang: str = "ru", batch_size: int = 10
) -> List[str]:
    """
    Translate a list of texts into the target language in batches.

    Args:
        texts (List[str]): List of original texts to translate.
        target_lang (str, optional): Target language code (e.g., 'ru'). Defaults to "ru".
        batch_size (int, optional): Number of texts per batch. Defaults to 50.

    Raises:
        ValueError: If the number of translated lines does not match the batch size.

    Returns:
        List[str]: List of translated texts in the same order.
    """
    results: List[str] = []
    delimiter = ";"
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        prompt = (
            f"Translate the following texts into {target_lang}, "
            f"Answer should contain only translated texts, "
            f"no additional information, preserving order and separated by '{delimiter}'"
            + "\n".join(batch)
        )
        translated = call_translate(prompt)
        print(translated)
        segments = [seg.strip() for seg in translated.split(delimiter)]
        if len(segments) != len(batch):
            raise ValueError(
                f"Expected {len(batch)} segments but got {len(segments)} translation results"
            )
        results.extend(segments)
    return results


def translate_dataset(
    dataset: DatasetDict,
    target_lang: str = "ru",
    batch_size: int = 10,
    splits: Optional[List[str]] = None,
) -> None:
    """
    Translate the 'tweet_cleaned' column in each split of a DatasetDict and save to CSV.

    Args:
        dataset (DatasetDict):
            A HuggingFace DatasetDict containing splits.
        target_lang (str, optional): Language code to translate into. Defaults to 'ru'.
        batch_size (int, optional): Batch size for API calls. Defaults to 50.
        splits (List[str], optional):
            List of splits to translate. Defaults to ['train', 'test', 'valid'].

    Returns:
        None: Saves translated CSV files for each split.
    """
    if splits is None:
        splits = ["train", "test", "valid"]

    for split in splits:
        ds = dataset[split]

        def translate_batch_fn(batch: dict) -> dict:
            originals = batch["tweet_cleaned"]
            batch_translated = batch_translate(
                originals, target_lang=target_lang, batch_size=batch_size
            )
            return {"translated_tweet": batch_translated}

        translated_ds = ds.map(translate_batch_fn, batched=True, batch_size=batch_size)
        df = translated_ds.to_pandas()
        df.to_csv(f"{split}_translated.csv", index=False)


def clean_tweets(data_dir: str, file_name: str) -> None:
    """
    Load tweets from a CSV,
    clean mentions and line breaks, remap classes, and save to a new CSV.

    Args:
        data_dir (str): Directory containing the input file.
        file_name (str): Name of the CSV file with raw tweets.

    Returns:
        None: Writes 'cleaned_tweets.csv' in the same directory.
    """
    input_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(input_path)
    df["tweet_cleaned"] = (
        df["tweet"]
        .str.replace(r"@[A-Za-z0-9]+\s*", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["class"] = df["class"].map({0: 0, 1: 0, 2: 1})
    output_path = os.path.join(data_dir, "cleaned_tweets.csv")
    df.to_csv(output_path, index=False)


def load_dataset(data_dir: str, file_name: str) -> DatasetDict:
    """
    Load tweets from a CSV file,
    create a DatasetDict from it, and split into train, test, and valid.

    Args:
        data_dir (str): Directory containing the CSV file.
        file_name (str): Name of the CSV file with tweets.

    Returns:
        DatasetDict: A dictionary of datasets for train, test, and valid splits.
    """
    df = pd.read_csv(os.path.join(data_dir, file_name))
    ds = Dataset.from_pandas(df)
    train_test_valid = ds.train_test_split(test_size=0.2)
    test_valid = train_test_valid["test"].train_test_split()
    train_test_valid_dataset = DatasetDict(
        {
            "train": train_test_valid["train"],
            "test": test_valid["test"],
            "valid": test_valid["train"],
        }
    )
    dataset = train_test_valid_dataset.remove_columns(
        ["hate_speech_count", "offensive_language_count", "neither_count", "count"]
    )
    return dataset


def classifier_train(cfg: DictConfig):
    """
    Train a sequence classification model
    """
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    wb_token = WANDB_API
    wandb.login(key=wb_token)
    wandb.init(
        project=cfg.wandb.project,
        job_type="training",
        config=OmegaConf.to_container(cfg, resolve=True),
        anonymous="allow",
    )

    file_path = cfg.data.data_path
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            data = [{"text": item[0], "label": item[1]} for item in raw_data]
    elif file_extension == ".csv":
        raw_data = pd.read_csv(file_path, encoding="utf-8")
        data = [
            {"text": row["tweet_cleaned"], "label": row["class"]}
            for _, row in raw_data.iterrows()
        ]
    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. Only .json and .csv are supported"
        )

    full_dataset = Dataset.from_list(data)
    class_proportions = sum(full_dataset["label"]) / len(full_dataset["label"])
    logging.info("Class Proportions: %s", class_proportions)
    split_dataset = full_dataset.train_test_split(
        test_size=cfg.data.eval_split, seed=cfg.data.seed
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name_or_path, num_labels=cfg.model.num_labels
    )

    # model.classifier.out_proj = nn.Linear(
    #     model.config.hidden_size, cfg.model.num_labels
    # )
    #
    # for name, param in model.named_parameters():
    #     if "classifier.out_proj" not in name:
    #         param.requires_grad = False
    for name, param in model.named_parameters():
        if "classifier" not in name:  # Freeze everything except the classifier head.
            param.requires_grad = False

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.tokenizer.max_length},
    )
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.tokenizer.max_length},
    )
    train_dataset = train_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
        }

    time_start = time.time()
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        # eval_strategy=cfg.training.evaluation_strategy,
        eval_steps=cfg.training.eval_steps
        if cfg.training.evaluation_strategy == "steps"
        else None,
        lr_scheduler_type="cosine",
        logging_dir=cfg.training.logging_dir,
        report_to="wandb",
        run_name=cfg.training.run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    logging.info("Training Time: %s", time.time() - time_start)
    results = trainer.evaluate()
    logging.info("Evaluation Results: %s", results)
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(os.path.join(cfg.training.output_dir, "tokenizer"))

    final_test_model(model, cfg.data.data_path, tokenizer, cfg.tokenizer.max_length)

    test_text = "Пример текста для оценки"
    inputs = tokenizer(
        test_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=cfg.tokenizer.max_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = outputs.logits.argmax(dim=-1).item()
    logging.info(f"Sample Prediction for '{test_text}': {prediction}")

    onnx_path = os.path.join(cfg.training.output_dir, "model.onnx")
    dummy_inputs = tokenizer(
        "This is a dummy input for ONNX export",
        return_tensors="pt",
        max_length=cfg.tokenizer.max_length,
        padding="max_length",
    )
    torch.onnx.export(
        model,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=cfg.onnx.opset_version,
    )

    downgraded = os.path.join(cfg.training.output_dir, "model_downgraded.onnx")
    model_onnx = onnx.load(onnx_path)
    model_onnx.ir_version = cfg.onnx.target_ir_version
    for imp in model_onnx.opset_import:
        if imp.domain in ("", "ai.onnx"):
            imp.version = cfg.onnx.target_opset_version
    onnx.checker.check_model(model_onnx)
    onnx.save(model_onnx, downgraded)
    logging.info(f"Downgraded ONNX saved to {downgraded}")

    logging.info(f"ONNX model exported to {onnx_path}")

    wandb.finish()


def test_trained_classifier(cfg: DictConfig, steps=720):
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(cfg.training.output_dir, f"checkpoint-{steps}")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(cfg.training.output_dir, "tokenizer")
    )
    test_text = "Пример текста для оценки"
    inputs = tokenizer(
        test_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=cfg.tokenizer.max_length,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = outputs.logits.argmax(dim=-1).item()
    logging.info(f"Sample Prediction for '{test_text}': {prediction}")
