install:
	pip install -r requirements/requirements.txt

install-dev: install
	pip install -r requirements/requirements_dev.txt
	pre-commit install