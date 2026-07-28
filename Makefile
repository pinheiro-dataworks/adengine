.PHONY: download pipeline train app test lint clean

download:
	python scripts/download_data.py --config configs/pipeline.yaml

pipeline:
	python -m adengine.cleaning --config configs/pipeline.yaml
	python -m adengine.features --config configs/pipeline.yaml
	python -m adengine.attribution --config configs/attribution.yaml --pipeline-config configs/pipeline.yaml

train:
	python -m adengine.segmentation --config configs/model.yaml --pipeline-config configs/pipeline.yaml
	python -m adengine.propensity   --config configs/model.yaml --pipeline-config configs/pipeline.yaml
	python -m adengine.metrics      --config configs/model.yaml --pipeline-config configs/pipeline.yaml
	python -m adengine.simulator    --config configs/model.yaml --pipeline-config configs/pipeline.yaml

app:
	streamlit run app/Home.py

test:
	pytest tests/ -v --cov=src/adengine --cov-report=term-missing

lint:
	ruff check src/ tests/ app/

clean:
	rm -rf data/staging/* data/marts/* models/*.joblib
