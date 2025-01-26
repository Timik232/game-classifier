import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Configuration
LMM_API_URL = "http://localhost:8000/v2/chat/completions"


class UserMessage(BaseModel):
    message: str


class ClassificationResponse(BaseModel):
    topic_class: str = None


class RelatedClassificATIONResponse(BaseModel):
    related_to_dnd: bool


def query_llm_api(prompt: str):
    """
    helper function for querying the LLM API
     :param prompt: message of the user
    """
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant"
                " specializing in classifying DND-related topics.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    response = requests.post(LMM_API_URL, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(
            status_code=response.status_code, detail="Error querying LLM API"
        )


@app.post("/check_dnd_relation", response_model=ClassificationResponse)
def check_dnd_relation(user_message: UserMessage):
    prompt = (
        f"Is the following message related to DND topics? "
        f"Respond with 'yes' or 'no': {user_message.message}"
    )
    llm_response = query_llm_api(prompt)

    response_content = (
        llm_response.get("choices")[0].get("message").get("content").strip().lower()
    )
    related_to_dnd = response_content == "yes"

    return ClassificationResponse(related_to_dnd=related_to_dnd)


@app.post("/get_dnd_topic_class", response_model=ClassificationResponse)
def get_dnd_topic_class(user_message: UserMessage):
    prompt = (
        f"Classify the following DND-related message into one of the classes: "
        f"mechanics, class, spell, race, bestiary, item, feats, backgrounds, inventory, lore. "
        f"If not related to DND, respond with 'unrelated'. Message: {user_message.message}"
    )
    llm_response = query_llm_api(prompt)

    response_content = (
        llm_response.get("choices")[0].get("message").get("content").strip().lower()
    )
    if response_content == "unrelated":
        return ClassificationResponse(related_to_dnd=False)

    return ClassificationResponse(related_to_dnd=True, topic_class=response_content)
