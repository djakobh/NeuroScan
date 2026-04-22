"""Downloads the neuroscan testing dataset from HF Hub, resolving LFS properly."""
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image

REPO_ID = "djakobh/neuroscan-testing"

files = [
    f for f in list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
print(f"Found {len(files)} images in dataset repo")

for repo_file in files:
    dest = Path("./data/Testing") / repo_file
    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=repo_file)
    shutil.copy2(cached, dest)

# Validate a sample to catch LFS pointer files early
samples = list(Path("./data/Testing").rglob("*.jpg"))[:3]
for sample in samples:
    try:
        img = Image.open(sample)
        img.verify()
        print(f"Image validation OK: {sample}")
    except Exception as e:
        print(f"ERROR: {sample} is not a valid image: {e}", file=sys.stderr)
        sys.exit(1)

print("Dataset download complete.")
