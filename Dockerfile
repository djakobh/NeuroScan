FROM python:3.11-slim
WORKDIR /app

COPY src/api/requirements.txt ./api-requirements.txt
COPY packages/ml/requirements.txt ./ml-requirements.txt
RUN pip install --no-cache-dir -r api-requirements.txt -r ml-requirements.txt huggingface_hub hf

COPY src/api/ ./src/api/
COPY packages/ml/ ./packages/ml/

RUN huggingface-cli download djakobh/neuroscan-testing --repo-type dataset --local-dir ./data/Testing

EXPOSE 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
