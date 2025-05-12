import argparse
import logging
import os
from typing import Dict, Iterator

import onnx
import torch
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
    quantize_static,
)
from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import (
    AutoCalibrationConfig,
    QuantFormat,
    QuantizationConfig,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class HFCalibrationDataReader(CalibrationDataReader):
    """
    Wraps a Hugging Face dataset for ONNX static quantization.

    Args:
        dataset (Dataset): Tokenized HF dataset with 'input_ids' and 'attention_mask'.
        tokenizer (PreTrainedTokenizer): Tokenizer to collate inputs.
        batch_size (int): Number of samples per calibration batch.
    """

    def __init__(self, dataset, tokenizer, batch_size: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.iterator = iter(self._get_batches())

    def _get_batches(self) -> Iterator[Dict[str, torch.Tensor]]:
        # Yield dicts matching ONNX inputs: {"input_ids": ..., "attention_mask": ...}
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i : i + self.batch_size]
            inputs = {
                "input_ids": torch.tensor(batch["input_ids"], dtype=torch.int64),
                "attention_mask": torch.tensor(
                    batch["attention_mask"], dtype=torch.int64
                ),
            }
            yield inputs

    def get_next(self) -> Dict[str, torch.Tensor]:
        try:
            return next(self.iterator)
        except StopIteration:
            return {}


def downgrade_onnx_versions(
    input_path: str,
    output_path: str,
    target_ir_version: int = 9,
    target_opset_version: int = 18,
) -> None:
    """
    Downgrades the IR and opset versions of an ONNX model.

    Args:
        input_path (str): Path to the source ONNX model file.
        output_path (str): Path where the downgraded ONNX model should be saved.
        target_ir_version (int): The desired IR version (e.g., 9).
        target_opset_version (int): The desired ONNX opset version (e.g., 18).

    Raises:
        onnx.checker.ValidationError: If the downgraded model fails validation.
    """
    # Load the original model from file
    model = onnx.load(input_path)

    # Set the IR version
    model.ir_version = target_ir_version

    # Update all opset imports to the target version
    for opset in model.opset_import:
        if opset.domain in ["", "ai.onnx"]:
            opset.version = target_opset_version

    # Save the modified model
    onnx.save(model, output_path)

    # Validate the saved model by file path to handle large protobufs
    try:
        onnx.checker.check_model(output_path)
        print(f"Model successfully saved and validated at {output_path}")
    except onnx.checker.ValidationError as e:
        print(f"Validation failed: {e}")
        raise


def export_model_to_onnx(dataset: Dataset | DatasetDict, cfg: DictConfig) -> None:
    """
    Exports a Hugging Face Transformer model to ONNX format, with optional downgrading.

    Args:
        dataset (Dataset):
        cfg (DictConfig): Configuration object with attributes:
            training.output_dir (str): Directory to save exported models.
            model.model_name_or_path (str): Pretrained model identifier or path.
            model.num_labels (int): Number of classification labels.
            tokenizer.max_length (int): Maximum token length.
            onnx.opset_version (int): Opset version for export.
            onnx.target_ir_version (int, optional): IR version for downgrading.
            onnx.target_opset_version (int, optional): Opset version for downgrading.

    Returns:
        None
    """
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.training.output_dir,
        num_labels=cfg.model.num_labels,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)

    # Ensure output directories exist
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    onnx_dir = os.path.join(cfg.training.output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    # Prepare dummy inputs
    dummy_inputs = tokenizer(
        "This is a dummy input for ONNX export",
        return_tensors="pt",
        max_length=cfg.tokenizer.max_length,
        padding="max_length",
        truncation=True,
    )

    onnx_path = os.path.join(onnx_dir, "model.onnx")

    # Export model to ONNX
    main_export(
        model_name_or_path=cfg.training.output_dir,
        output=onnx_dir,  # Exports model.onnx and other files to this directory
        task="text-classification",
        opset=cfg.onnx.opset_version,
        tokenizer=tokenizer,
        do_validation=True,  # Validates the exported model
    )

    logging.info(f"ONNX model exported to {onnx_path}")

    tokenized = dataset.map(
        lambda ex: tokenizer(
            ex["tweet_cleaned"],
            truncation=True,
            padding="max_length",
            max_length=cfg.tokenizer.max_length,
        ),
        batched=True,
        remove_columns=dataset.column_names,
    )
    calib_ds = tokenized.select(range(cfg.data.calib_samples))

    # 3) Initialize Quantizer
    quantizer = ORTQuantizer.from_pretrained(onnx_dir)

    # 4) Calibration config (e.g. Percentile method)
    calib_config = AutoCalibrationConfig.percentiles(calib_ds, percentile=99.9)

    # 5) Run fit (produces calibration ranges)
    quantizer.fit(
        dataset=calib_ds,
        calibration_config=calib_config,
    )

    # 6) Define static quantization config
    qconfig = QuantizationConfig(
        is_static=True,
        format=QuantFormat.QDQ,
        per_channel=True,
        activations_dtype=QuantType.QInt8,
        weights_dtype=QuantType.QInt8,
    )

    # 7) Quantize with saved ranges
    quantizer.quantize(
        save_dir=os.path.join(onnx_dir, "quantized"),
        quantization_config=qconfig,
        file_suffix="static",
    )
    # Optionally downgrade IR and opset
    # downgraded_path = os.path.join(onnx_dir, "model_downgraded.onnx")
    # downgrade_onnx_versions(
    #     onnx_quant,
    #     downgraded_path,
    #     cfg.onnx.target_ir_version,
    #     cfg.onnx.target_opset_version,
    # )
    # logging.info(f"ONNX model downgraded to {downgraded_path}")


def quantize_onnx_static(
    onnx_input: str, onnx_output: str, calibration_reader: CalibrationDataReader
) -> None:
    """
    Apply static quantization to an ONNX model file with calibration.

    Args:
        onnx_input (str): Path to floating-point ONNX.
        onnx_output (str): Path to save quantized ONNX.
        calibration_reader (CalibrationDataReader): Supplies calibration data.
    """
    quantize_static(
        model_input=onnx_input,
        model_output=onnx_output,
        calibration_data_reader=calibration_reader,
        quant_format=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )


def argparser():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Downgrade ONNX model versions.")

    # Add arguments for input and output model paths
    parser.add_argument("input_model", type=str, help="Path to the input ONNX model")
    parser.add_argument(
        "output_model", type=str, help="Path to save the downgraded ONNX model"
    )

    # Optional arguments for IR version and Opset version
    parser.add_argument(
        "--target_ir_version",
        type=int,
        default=9,
        help="Target IR version (default: 8)",
    )
    parser.add_argument(
        "--target_opset_version",
        type=int,
        default=17,
        help="Target Opset version (default: 17)",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Execute version downgrade
    downgrade_onnx_versions(
        args.input_model,
        args.output_model,
        args.target_ir_version,
        args.target_opset_version,
    )
