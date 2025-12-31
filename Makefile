run:
	uvicorn app.main:app --reload

test:
	pytest -q

docker-build:
	docker build -t llm-exchange:latest .
