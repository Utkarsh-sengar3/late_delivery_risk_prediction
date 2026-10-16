install:
	python -m pip install -r requirements.txt

run:
	streamlit run app/Home.py

test:
	pytest -q
