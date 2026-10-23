PYTHON ?= .venv/bin/python
DATA ?= data/raw/Dataset Renovacion_prestamo.csv
API_PORT ?= 8000
MLFLOW_PORT ?= 5000
MLFLOW_HOSTS = localhost:*,127.0.0.1:*
ifdef CODESPACE_NAME
MLFLOW_HOSTS = localhost:*,127.0.0.1:*,$(CODESPACE_NAME)-$(MLFLOW_PORT).app.github.dev
endif
.PHONY: install test check-data pipeline notebook api mlflow docker dvc
install:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install -e '.[dev,api,mlops]'
	$(PYTHON) -m ipykernel install --user --name loan-renewal --display-name 'Loan renewal (.venv)'
test:
	$(PYTHON) -m unittest discover -s tests -v
check-data:
	$(PYTHON) -m loan_renewal version-data --data "$(DATA)"
pipeline: check-data
	$(PYTHON) pipelines/run_training.py --data "$(DATA)" --tune --track
notebook: check-data
	$(PYTHON) scripts/execute_notebook.py notebooks/01_renovacion_prestamo.ipynb
api:
	$(PYTHON) -m uvicorn api.app:app --host 127.0.0.1 --port $(API_PORT)
mlflow:
	$(PYTHON) -m mlflow ui --backend-store-uri "sqlite:///$(CURDIR)/mlruns/mlflow.db" --host 127.0.0.1 --port $(MLFLOW_PORT) --workers 1 --allowed-hosts "$(MLFLOW_HOSTS)"
docker:
	docker compose up --build --wait -d
dvc: check-data
	PATH="$(CURDIR)/.venv/bin:$$PATH" $(PYTHON) -m dvc repro
