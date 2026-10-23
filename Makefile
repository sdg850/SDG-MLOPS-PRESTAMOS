PYTHON ?= .venv/bin/python
.PHONY: install test pipeline notebook api docker dvc
install:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install -e '.[dev,api,mlops]'
	$(PYTHON) -m ipykernel install --user --name loan-renewal --display-name 'Loan renewal (.venv)'
test:
	$(PYTHON) -m unittest discover -s tests -v
pipeline:
	$(PYTHON) pipelines/run_training.py --tune --track
notebook:
	$(PYTHON) scripts/execute_notebook.py notebooks/01_renovacion_prestamo.ipynb
api:
	$(PYTHON) -m uvicorn api.app:app --host 127.0.0.1 --port 8000
docker:
	docker compose up --build --wait -d
dvc:
	PATH="$(CURDIR)/.venv/bin:$$PATH" $(PYTHON) -m dvc repro
