# Topic Classifier

# Build
For using this service, you need to run llm
service with the same network as this service. Replace network in the
docker-compose file for appropriate.
Also it may be needed to change url to the service in the 
`utils.py` file.
```bash
docker-compose build
```
# Run
```bash
docker-compose up
```
# Endpoints
For first mentioned endpoint I will write path with the assumption that the server is running on localhost:8000,
because it's base port for docker-compose. Then I will write only the path to the endpoint.
For all endpoints, exept test, http request should include one field: `messages: str`
- `http://localhost:8001/` — connection test
- `/get_dnd_topic_class` — get topic of the message
- `/check_dnd_relation` — check if the message is related to the dnd subject
