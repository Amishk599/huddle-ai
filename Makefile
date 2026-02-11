.PHONY: setup install init-stores run test clean

setup:
	python3 -m venv venv
	@echo "Virtual environment created. Run 'source venv/bin/activate'"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

init-stores:
	python3 scripts/setup_vector_stores.py

run:
	python3 -m src.main

test:
	pytest tests/ -v

eval:
	python3 -m evals.run_evals

eval-meeting:
	python3 -m evals.run_evals --meeting

eval-assistant:
	python3 -m evals.run_evals --assistant

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf data/vector_stores/*/
	rm -f emails_sent/*.txt

demo:
	python3 scripts/generate_demo_traces.py
