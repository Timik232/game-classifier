FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-root
COPY . .
CMD ["poetry", "run", "uvicorn", "llmserver.gpt4free_server:app", "--host", "0.0.0.0", "--port", "8001"]
