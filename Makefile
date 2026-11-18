PYTHON ?= .venv/bin/python
RUN_DIR ?= data/results/reproduction

.PHONY: test app reproduce report

test:
	$(PYTHON) -m pytest -q -W error

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501

reproduce:
	$(PYTHON) -m src.experiment --snapshot data/raw/full_ten_stocks_corrected --output $(RUN_DIR)

report:
	$(PYTHON) scripts/build_report.py --results app/artifacts/full_walkforward --output output
