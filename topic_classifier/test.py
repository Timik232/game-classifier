import logging
import time

import requests

from .utils import (
    BASE_URL_LLM,
    BASE_URL_MAIN,
    CHECK_DND_RELATION,
    DND_TOPIC_CLASS,
    REQUEST_TIMEOUT,
    TEST_DELAY,
)


def log_test_start(func):
    """Decorator to log test start/end with timing"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Starting test: {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(
            f"Completed test: {func.__name__} in {time.time() - start_time:.2f}s"
        )
        return result

    return wrapper


def make_request(
    url: str, method: str = "GET", payload: dict = None
) -> requests.Response:
    """Helper function to handle HTTP requests with error handling"""
    try:
        if method == "POST":
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        else:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)

        response.raise_for_status()
        logging.debug(f"Response: {response.json()}")
        return response

    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed to {url}: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            logging.error(f"Response content: {e.response.text}")
        raise


@log_test_start
def test_root():
    """Test if the main server is responding"""
    response = make_request(f"{BASE_URL_MAIN}/")
    assert response.status_code == 200, "Unexpected status code"
    assert "message" in response.json(), "Missing 'message' in response"
    logging.info("Main server is operational")


@log_test_start
def test_another_app():
    """Test if the LLM server is available"""
    response = make_request(f"{BASE_URL_LLM}/")
    assert response.status_code == 200, "Unexpected status code"
    assert "message" in response.json(), "Missing 'message' in response"
    logging.info("LLM server is operational")


@log_test_start
def test_check_dnd_relation():
    """Test DND relation detection endpoint"""
    payload = {"messages": "Я люблю ДНД!"}
    response = make_request(CHECK_DND_RELATION, "POST", payload)

    assert "related_to_dnd" in response.json(), "Missing classification field"
    logging.debug("DND relation check response: %s", response.json())
    logging.info("DND relation detection working")


@log_test_start
def test_get_dnd_topic_class():
    """Test topic classification for class-related queries"""
    payload = {"messages": "Расскажи про класс воина"}

    # First verify the message is DND-related
    relation_response = make_request(CHECK_DND_RELATION, "POST", payload)
    assert (
        relation_response.json()["related_to_dnd"] is True
    ), "Message should be DND-related"

    # Then test topic classification
    class_response = make_request(DND_TOPIC_CLASS, "POST", payload)
    assert "topic_class" in class_response.json(), "Missing topic classification"
    assert (
        class_response.json()["topic_class"] == "class"
    ), "Unexpected topic classification"
    logging.info("Class topic classification working")


def _test_topic_case(message: str, expected_topic: str) -> None:
    """Helper function to test different topic classifications"""
    payload = {"messages": message}

    # Verify DND relation
    relation_response = make_request(CHECK_DND_RELATION, "POST", payload)
    assert (
        relation_response.json()["related_to_dnd"] is True
    ), f"Message `{payload['messages']}` should be DND-related"

    # Test topic classification
    class_response = make_request(DND_TOPIC_CLASS, "POST", payload)
    assert class_response.status_code == 200, "Unexpected status code"
    predicted_class = class_response.json().get("topic_class")
    assert (
        predicted_class == expected_topic
    ), f"Expected {expected_topic} classification, get {predicted_class} instead"


@log_test_start
def test_spell_case():
    """Test spell-related topic classification"""
    _test_topic_case(
        "мой маг 5 уровня изучает заклинание огненный шар, расскажи про него", "spell"
    )
    logging.info("Spell classification working")


@log_test_start
def test_race_case():
    """Test race-related topic classification"""
    _test_topic_case("дварф", "race")
    logging.info("Race classification working")


@log_test_start
def test_mechanics_case():
    """Test game mechanics classification"""
    _test_topic_case("Как работает проверка ловкости?", "mechanics")
    logging.info("Mechanics classification working")


@log_test_start
def test_barbarian_case():
    """Test game mechanics classification"""
    _test_topic_case("почему варвары такие злые?", "class")
    logging.info("Mechanics classification working")


@log_test_start
def test_barbarian2_case():
    """Test game mechanics classification"""
    _test_topic_case("расскажи мне о характеристиках варвара", "class")
    logging.info("Mechanics classification working")


@log_test_start
def test_unicorn_case():
    """Test game mechanics classification"""
    _test_topic_case("сколько весит единорог?", "bestiary")
    logging.info("Mechanics classification working")


def full_test():
    """Execute all tests with proper sequencing and reporting"""
    logging.info("Starting comprehensive test suite")

    tests = [
        test_root,
        test_another_app,
        test_check_dnd_relation,
        test_get_dnd_topic_class,
        test_spell_case,
        test_race_case,
        test_mechanics_case,
        test_barbarian_case,
        test_barbarian2_case,
        test_unicorn_case,
    ]

    results = {"passed": 0, "failed": 0, "errors": []}

    for test in tests:
        try:
            test()
            time.sleep(TEST_DELAY)
            results["passed"] += 1
        except AssertionError as e:
            logging.error(f"Test failed: {test.__name__} - {str(e)}")
            results["failed"] += 1
            results["errors"].append((test.__name__, str(e)))
        except Exception as e:
            logging.error(f"Unexpected error in {test.__name__}: {str(e)}")
            results["failed"] += 1
            results["errors"].append((test.__name__, str(e)))

    logging.info("\n=== Test Summary ===")
    logging.info(f"Total tests: {len(tests)}")
    logging.info(f"Passed: {results['passed']}")
    logging.info(f"Failed: {results['failed']}")

    if results["errors"]:
        logging.info("\nError Details:")
        for test_name, error in results["errors"]:
            logging.info(f"- {test_name}: {error}")

    logging.info(f"Success rate: {results['passed'] / len(tests) * 100:.1f}%.")
