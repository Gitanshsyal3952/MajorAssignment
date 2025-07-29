FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src

# Run train.py before predict.py so model.joblib gets created
CMD ["sh", "-c", "python src/train.py && python src/predict.py"]
