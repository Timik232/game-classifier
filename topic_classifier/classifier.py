import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .utils import CLASS_PROMPT, RELATION_PROMPT

app = FastAPI()

# Configuration
LMM_API_URL = "http://fastapi-server:8000/v2/chat/completions"


class UserMessage(BaseModel):
    messages: str


class ClassificationResponse(BaseModel):
    topic_class: str = None


class RelatedClassificationResponse(BaseModel):
    related_to_dnd: bool


def query_llm_api(prompt: str) -> dict:
    """
    helper function for querying the LLM API
     :param prompt: message of the user
    """
    try:
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "Вы являетесь ассистентом, "
                    "специализирующимся на классификации тем, связанных с DND.",
                },
                {"role": "user", "content": prompt},
            ],
        }

        response = requests.post(LMM_API_URL, json=payload)
        response.raise_for_status()
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code, detail="Error querying LLM API"
            )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"LLM API unreachable: {str(e)}")


def check_answer_step_by_step(llm_response) -> str:
    """
    check that answer is valid
    :param llm_response:
    :return: final message from the answer
    """
    choices = llm_response.get("choices")
    if not choices:
        raise HTTPException(status_code=502, detail="No choices in LLM response")

    first_choice = choices[0]
    message = first_choice.get("message")
    if not message:
        raise HTTPException(status_code=502, detail="No message in LLM response")

    content = message.get("content")
    if not content:
        raise HTTPException(status_code=502, detail="No content in LLM response")

    response_content = content.strip().lower()
    return response_content


def check_classes(message: str) -> str:
    """
    check if the format of the answer, is it have only one class
    :param message: answer from the model
    :return: class if it's ok or unrelated if the answer is invalid
    """
    valid_classes = [
        "mechanics",
        "class",
        "spell",
        "race",
        "bestiary",
        "item",
        "feats",
        "backgrounds",
        "inventory",
        "lore",
    ]
    return message if message in valid_classes else "unrelated"


@app.post("/check_dnd_relation", response_model=RelatedClassificationResponse)
def check_dnd_relation(user_message: UserMessage):
    """
    dnd relation validation
    :param user_message: http request
    :return: boolean value if the message is related to DND
    """
    prompt = RELATION_PROMPT + user_message.messages
    llm_response = query_llm_api(prompt)
    response_content = check_answer_step_by_step(llm_response)
    if "yes" in response_content:
        related_to_dnd = True
    elif "no" in response_content:
        related_to_dnd = False
    else:
        related_to_dnd = False
    return RelatedClassificationResponse(related_to_dnd=related_to_dnd)


@app.post("/get_dnd_topic_class", response_model=ClassificationResponse)
def get_dnd_topic_class(user_message: UserMessage):
    """
    dnd topic classification
    :param user_message: http request
    :return: topic of the dnd
    """
    prompt = CLASS_PROMPT + user_message.messages
    llm_response = query_llm_api(prompt)
    response_content = check_answer_step_by_step(llm_response)
    dnd_class = check_classes(response_content)
    return ClassificationResponse(topic_class=dnd_class)


@app.get("/")
async def root():
    """
    check server availability
    """
    return {
        "message": "Server is running. Use "
        "the /check_dnd_relation and /get_dnd_topic_clas endpoints "
        "to interact."
    }
