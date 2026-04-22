FROM python:3.11-slim
WORKDIR /app

COPY src/api/requirements.txt ./api-requirements.txt
COPY packages/ml/requirements.txt ./ml-requirements.txt
RUN pip install --no-cache-dir -r api-requirements.txt -r ml-requirements.txt huggingface_hub

COPY src/api/ ./src/api/
COPY packages/ml/ ./packages/ml/

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='djakobh/neuroscan-testing', repo_type='dataset', local_dir='./data/Testing')"

EXPOSE 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
