FROM python:3.11-slim
WORKDIR /app

COPY src/api/requirements.txt ./api-requirements.txt
COPY packages/ml/requirements.txt ./ml-requirements.txt
RUN pip install --no-cache-dir -r api-requirements.txt -r ml-requirements.txt huggingface_hub && \
    pip install --no-cache-dir opencv-python-headless --force-reinstall

COPY src/api/ ./src/api/
COPY packages/ml/ ./packages/ml/
COPY download_dataset.py ./download_dataset.py

RUN python download_dataset.py

EXPOSE 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
