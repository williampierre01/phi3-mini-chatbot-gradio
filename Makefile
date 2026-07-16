
.PHONY: install run benchmark sweep clean
 
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
 
run:
	python app.py
 
benchmark:
	python benchmark.py --output benchmark_results.csv
 
sweep:
	python benchmark.py --sweep-batch --output benchmark_results.csv
 
clean:
	rm -rf ~/.cache/huggingface/hub/*
	@echo "Cache do Hugging Face limpo com sucesso."
 