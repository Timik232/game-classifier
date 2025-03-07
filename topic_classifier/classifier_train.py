import json
import logging
import os

import hydra
import torch
import torch.nn as nn
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import wandb

from .private_api import WANDB_API


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def classifier_train(cfg: DictConfig):
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    wb_token = WANDB_API

    wandb.login(key=wb_token)
    wandb.init(
        project=cfg.wandb.project,
        job_type="training",
        config=OmegaConf.to_container(cfg, resolve=True),
        anonymous="allow",
    )

    with open(cfg.data.data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data = [{"text": item[0], "label": item[1]} for item in raw_data]

    full_dataset = Dataset.from_list(data)

    split_dataset = full_dataset.train_test_split(
        test_size=cfg.data.eval_split, seed=cfg.data.seed
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
    model = AutoModelForSequenceClassification.from_pretrained(
        "intfloat/multilingual-e5-large-instruct", num_labels=cfg.model.num_labels
    )

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.tokenizer.max_length,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # for name, param in model.named_parameters():
    #     logging.info(f"{name}: {param}")

    model.classifier = nn.Linear(model.config.hidden_size, cfg.model.num_labels)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        evaluation_strategy=cfg.training.evaluation_strategy,
        eval_steps=cfg.training.eval_steps
        if cfg.training.evaluation_strategy == "steps"
        else None,
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
    results = trainer.evaluate()
    logging.info("Evaluation Results:", results)

    trainer.save_model(cfg.training.output_dir)

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

    wandb.finish()


if __name__ == "__main__":
    classifier_train()
