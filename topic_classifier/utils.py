# prompts
RELATION_PROMPT = (
    "Есть сообщение пользователя. Думай по шагам и оцени, что пользователь "
    "хочет узнать, и относится ли его сообщение к теме DND? "
    "Категории, которые существуют: "
    "mechanics, class, spell, race, bestiary, item, feats, "
    "backgrounds, inventory, lore. "
    "Ответьте в формате:\n"
    "Почему я думаю, что пользователь спросил именно это:\n"
    "Пользователь хотел спросить:\n"
    "Вердикт:\n"
    "Вердикт может содержать только 'yes' или 'no'. yes если относится к dnd, "
    "no если не относится.\n Пользователь: "
)

CLASS_PROMPT = (
    "Классифицируйте сообщение на русском языке в одну из английских категорий: "
    "mechanics, class, spell, race, bestiary, item, feats, "
    "backgrounds, inventory, lore. Ответ должен "
    "содержать только название категории и больше ничего. "
    "Сообщение: "
)

# Constants
BASE_URL_MAIN = "http://localhost:8001"
BASE_URL_LLM = "http://localhost:8000"
CHECK_DND_RELATION = f"{BASE_URL_MAIN}/check_dnd_relation"
DND_TOPIC_CLASS = f"{BASE_URL_MAIN}/get_dnd_topic_class"
REQUEST_TIMEOUT = 15  # seconds
TEST_DELAY = 5  # seconds between tests
