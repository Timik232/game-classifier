import json
import logging
import os

import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score
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
):
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
    class_proportions = sum(full_dataset["label"]) / len(full_dataset["label"])
    logging.info("Class Proportions: %s", class_proportions)
    split_dataset = full_dataset.train_test_split(
        test_size=cfg.data.eval_split, seed=cfg.data.seed
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
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

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        eval_strategy=cfg.training.evaluation_strategy,
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
    results = trainer.evaluate()
    logging.info("Evaluation Results: %s", results)
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(os.path.join(cfg.training.output_dir), "tokenizer")

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
