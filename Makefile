install: 
	sudo apt-get update
	sudo apt-get install python3

init:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

run: init
	.venv/bin/python3 project/__main__.py

clean:
	rm -rf project/algorithms/commons/__pycache__
	rm -rf project/src/services/__pycache__
	rm -rf project/algorithms/__pycache__
	rm -rf project/preprocess/__pycache__
	rm -rf project/src/utils/__pycache__
	rm -rf project/src/app/__pycache__
	rm -rf project/src/__pycache__
	rm -rf .venv

.PHONY: run
