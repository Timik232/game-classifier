import requests


def test_root():
    response = requests.get("http://localhost:8001/")
    assert response.status_code == 200
    assert "message" in response.json()
    print("Root endpoint test passed.")


def test_check_dnd_relation():
    payload = {"message": "Я люблю ДНД!"}
    response = requests.post("http://localhost:8001/check_dnd_relation", json=payload)
    print(response)
    assert response.status_code == 200
    assert "related_to_dnd" in response.json()
    response_data = response.json()
    assert response_data["related_to_dnd"]
    print("Check DND relation endpoint test passed.")


def test_get_dnd_topic_class():
    payload = {"message": "Расскажи про класс воина"}
    response = requests.post("http://localhost:8001/get_dnd_topic_class", json=payload)
    assert response.status_code == 200
    assert "topic_class" in response.json()
    print("Get DND topic class endpoint test passed.")


def full_test():
    print("Testing FastAPI server...")
    test_root()
    test_check_dnd_relation()
    test_get_dnd_topic_class()
    print("All tests passed.")


if __name__ == "__main__":
    full_test()
