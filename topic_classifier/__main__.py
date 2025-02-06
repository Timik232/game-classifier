import uvicorn


def main():
    uvicorn.run(
        "topic_classifier.classifier:app", host="0.0.0.0", port=8001, reload=False
    )


if __name__ == "__main__":
    main()
