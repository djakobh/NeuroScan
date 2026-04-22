FROM python:3.11-slim
WORKDIR /app

COPY src/api/requirements.txt ./api-requirements.txt
COPY packages/ml/requirements.txt ./ml-requirements.txt
RUN pip install --no-cache-dir -r api-requirements.txt -r ml-requirements.txt huggingface_hub && \
    pip install --no-cache-dir opencv-python-headless --force-reinstall

COPY src/api/ ./src/api/
COPY packages/ml/ ./packages/ml/

RUN python -c "\
from huggingface_hub import hf_hub_download, list_repo_files; \
import shutil, sys; \
from pathlib import Path; \
from PIL import Image; \
repo_id = 'djakobh/neuroscan-testing'; \
files = [f for f in list_repo_files(repo_id=repo_id, repo_type='dataset') \
         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]; \
print(f'Found {len(files)} images in dataset repo'); \
for repo_file in files: \
    dest = Path('./data/Testing') / repo_file; \
    dest.parent.mkdir(parents=True, exist_ok=True); \
    cached = hf_hub_download(repo_id=repo_id, repo_type='dataset', filename=repo_file); \
    shutil.copy2(cached, dest); \
sample = list(Path('./data/Testing').rglob('*.jpg'))[:1]; \
if sample: \
    try: Image.open(sample[0]).verify(); print(f'Image validation OK: {sample[0]}'); \
    except Exception as e: print(f'ERROR: images are not valid JPEGs: {e}', file=sys.stderr); sys.exit(1); \
"

EXPOSE 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
