import logging

from topic_classifier import configure_logging, full_test

if __name__ == "__main__":
    configure_logging(logging.DEBUG)
    full_test()
