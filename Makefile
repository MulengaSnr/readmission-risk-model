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

.PHONY: data eda baselines models evaluate explain fairness
data:    ## Download, clean and split the dataset
	uv run python -m readmission_risk.data.download
	uv run python -m readmission_risk.data.clean
	uv run python -m readmission_risk.data.split

eda:     ## Regenerate the EDA notebook (from its paired .py) and figures
	uv run jupytext --to ipynb notebooks/01_eda.py
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

baselines: ## Grouped cross-validation of the baseline models (training set only)
	uv run python -m readmission_risk.models.baselines

models:  ## Tune random forest + XGBoost (one process each), then compare (~45 min)
	uv run python -m readmission_risk.models.tuning --tune random_forest
	uv run python -m readmission_risk.models.tuning --tune xgboost
	uv run python -m readmission_risk.models.tuning --compare

evaluate: ## One-time test-set evaluation of the final models (refuses to overwrite)
	uv run python -m readmission_risk.models.evaluate

explain: ## SHAP importance and dependence for the final model (test set)
	uv run python -m readmission_risk.models.explain

fairness: ## Fairness audit by sex, age band and race (test set)
	uv run python -m readmission_risk.models.fairness
