FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-root
COPY ./topic_classifier ./topic_classifier
#CMD ["poetry", "run", "uvicorn", "topic_classifier.classifier:app", "--host", "0.0.0.0", "--port", "8001"]
CMD ["python", "-m", "topic_classifier"]
