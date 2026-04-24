.PHONY: install generate train run run-api run-frontend run-all test lint format \
        docker-build docker-run docker-run-api docker-run-frontend clean

install:
	pip install -r requirements.txt -r requirements-dev.txt

generate:
	python data/generate_synthetic_data.py

train:
	python pipeline.py --stage all

# ─── Run targets ──────────────────────────────────────────────────────────────
run:
	streamlit run app.py

run-api:
	uvicorn api:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
	cd frontend && npm run dev

run-all:
	@echo "Start the API in one terminal: make run-api"
	@echo "Start the frontend in another: make run-frontend"

# ─── Frontend setup ───────────────────────────────────────────────────────────
frontend-install:
	cd frontend && npm install

# ─── Tests ────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

lint:
	ruff check . && mypy . --ignore-missing-imports

format:
	black . && ruff check . --fix

# ─── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -t eyewear-platform:latest .
	docker build -f Dockerfile.api -t eyewear-api:latest .
	docker build -f frontend/Dockerfile.frontend -t eyewear-frontend:latest ./frontend

docker-run:
	docker-compose up -d

docker-run-api:
	docker-compose up -d eyewear-api

docker-run-frontend:
	docker-compose up -d eyewear-frontend

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pkl" -delete; \
	rm -rf data/synthetic/*.csv models/artifacts/*
