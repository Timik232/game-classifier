import logging

import hydra
from omegaconf import omegaconf

from topic_classifier import (
    classifier_train,
    clean_tweets,
    configure_logging,
    full_test,
    load_dataset,
    translate_dataset,
)


def prepate_data(cfg: omegaconf.DictConfig) -> None:
    clean_tweets(cfg.data.dir, cfg.data.file_name)
    dataset = load_dataset(cfg.data.dir, "cleaned_tweets.csv")
    translate_dataset(dataset)


@hydra.main(
    version_base="1.1", config_path="topic_classifier/conf", config_name="config"
)
def main(cfg: omegaconf.DictConfig):
    classifier_train(cfg)


if __name__ == "__main__":
    configure_logging(logging.DEBUG)
    # full_test()
    main()
