FROM python:3.10-slim

WORKDIR /app

COPY flask_app/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

COPY models/label_encoder.pkl /app/models/label_encoder.pkl

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5001

#local
CMD ["python", "app.py"]  

#Production
# CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--timeout", "120", "app:app"]