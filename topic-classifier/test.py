import requests


def test_root():
    """
    test if the server is enabled
    """
    response = requests.get("http://localhost:8001/")
    assert response.status_code == 200
    assert "message" in response.json()
    print("Root endpoint test passed.")


def test_another_app():
    """
    test if the base server with llm is enable
    :return:
    """
    response = requests.get("http://localhost:8000/")
    assert response.status_code == 200
    assert "message" in response.json()
    print("Another app test passed.")


def test_check_dnd_relation():
    payload = {"messages": "Я люблю ДНД!"}
    response = requests.post("http://localhost:8001/check_dnd_relation", json=payload)
    print(response)
    assert response.status_code == 200
    assert "related_to_dnd" in response.json()
    print("Check DND relation endpoint test passed.")


def test_get_dnd_topic_class():
    payload = {"messages": "Расскажи про класс воина"}
    response = requests.post("http://localhost:8001/get_dnd_topic_class", json=payload)
    assert response.status_code == 200
    assert "topic_class" in response.json()
    assert response.json()["topic_class"] == "class"
    print("Get DND topic class endpoint test passed.")


def test_spell_case():
    payload = {
        "messages": "мой маг 5 уровня изучает заклинание огненный шар, расскажи про него"
    }
    response = requests.post("http://localhost:8001/get_dnd_topic_class", json=payload)
    assert response.status_code == 200
    assert response.json()["topic_class"] == "spell"
    print("Spell topic test passed.")


def test_race_case():
    payload = {"messages": "дварф"}
    response = requests.post("http://localhost:8001/get_dnd_topic_class", json=payload)
    assert response.status_code == 200
    assert response.json()["topic_class"] == "race"
    print("Race topic test passed.")


def test_mechanics_case():
    payload = {"messages": "Как работает проверка ловкости?"}
    response = requests.post("http://localhost:8001/get_dnd_topic_class", json=payload)
    assert response.status_code == 200
    assert response.json()["topic_class"] == "mechanics"
    print("Mechanic topic test passed.")


def full_test():
    """
    run all tests one by one
    """
    print("Testing FastAPI server...")
    test_root()
    test_another_app()
    test_get_dnd_topic_class()
    test_check_dnd_relation()
    test_spell_case()
    test_race_case()
    test_mechanics_case()
    print("All tests passed.")


if __name__ == "__main__":
    full_test()
