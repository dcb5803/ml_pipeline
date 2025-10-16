FROM python:3.10-slim
WORKDIR /app
COPY ml_pipeline.py .
RUN pip install pandas numpy scikit-learn fastapi uvicorn mlflow
EXPOSE 8000
CMD ["python", "ml_pipeline.py"]
