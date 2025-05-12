import logging
from typing import List, Optional

from datasets import DatasetDict
from libretranslatepy import LibreTranslateAPI
from tenacity import retry, stop_after_attempt, wait_exponential

from .utils import TRANSLATE_URL  # you can repurpose LMSTUDIO_URL as your LT endpoint

# Initialize the LibreTranslate client
lt_client = LibreTranslateAPI(
    url=TRANSLATE_URL,  # e.g. "http://localhost:5000/"
    api_key="<YOUR_LT_KEY>",  # only if you’ve enabled API‐keys; otherwise omit
)


@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_translate(
    text: str, source_lang: str = "auto", target_lang: str = "ru"
) -> str:
    """
    Call LibreTranslate API to translate a given text.

    Args:
        text (str): The text to translate.
        source_lang (str): Source language code (ISO 639), 'auto' for auto-detection.
        target_lang (str): Target language code (ISO 639).

    Returns:
        str: The translated text.
    """
    # This line makes the HTTP POST under the hood:
    translated_text = lt_client.translate(
        q=text, source=source_lang, target=target_lang
    )
    return translated_text.strip()


@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(6))
def translate_single(text: str, target_lang: str = "ru") -> str:
    """
    Translate a single text into the target language.

    Args:
        text (str): Text to translate.
        target_lang (str): Target language code (e.g., 'ru').

    Returns:
        str: Translated text.
    """
    # LibreTranslate auto-detects source if you pass 'auto'
    return call_translate(text, source_lang="auto", target_lang=target_lang)


def translate_dataset(
    dataset: DatasetDict,
    target_lang: str = "ru",
    splits: Optional[List[str]] = None,
) -> None:
    """
    Translate the 'tweet_cleaned' column one sample at a time and
    save each split to a CSV file.

    Args:
        dataset (DatasetDict): HuggingFace DatasetDict to translate.
        target_lang (str): Target language code.
        splits (List[str], optional): List of splits to process.
            Defaults to ['train', 'test', 'valid'].
    """
    if splits is None:
        splits = ["train", "test", "valid"]

    for split in splits:
        ds = dataset[split]

        def translate_example_fn(example: dict) -> dict:
            """Translate one example at a time."""
            try:
                translated = translate_single(example["tweet_cleaned"], target_lang)
                return {"translated_tweet": translated}
            except Exception as e:
                logging.error(
                    f"Error translating sample: {example['tweet_cleaned']} - {e}"
                )
                return {"translated_tweet": ""}

        # Map sample by sample
        translated_ds = ds.map(
            translate_example_fn, batched=False, desc=f"Translating {split} set"
        )

        # Dump to CSV
        df = translated_ds.to_pandas()
        df.to_csv(f"{split}_translated.csv", index=False)
        logging.info(f"Saved translated {split} set with {len(df)} samples")
