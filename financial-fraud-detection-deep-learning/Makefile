.PHONY: install data train test api app docker triton clean

install:
	python -m pip install -e . -r requirements-dev.txt

data:
	python scripts/generate_sample_data.py

train:
	python scripts/train_all.py

test:
	pytest -q

api:
	uvicorn api.main:app --reload

app:
	streamlit run app/streamlit_app.py

docker:
	docker compose up --build

triton:
	python scripts/export_triton.py

clean:
	rm -rf data/processed/* artifacts/predictions/* reports/*.png reports/model_metrics.json
