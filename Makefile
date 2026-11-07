.PHONY: setup lint format test check data eda

setup:   ## Create the virtual environment and install dependencies
	uv sync

lint:    ## Static checks (PEP 8, imports, docstrings, common bugs)
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix formatting and safe lint issues
	uv run ruff format .
	uv run ruff check --fix .

test:    ## Run the test suite with coverage
	uv run pytest --cov=src --cov-report=term-missing

check: lint test  ## Everything CI runs

.PHONY: data eda
data:    ## Download, clean and split the dataset
	uv run python -m readmission_risk.data.download
	uv run python -m readmission_risk.data.clean
	uv run python -m readmission_risk.data.split

eda:     ## Regenerate the EDA notebook (from its paired .py) and figures
	uv run jupytext --to ipynb notebooks/01_eda.py
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
