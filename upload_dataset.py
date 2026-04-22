"""
One-time script to upload data/Testing/ to a HF Dataset repo.
Run: python upload_dataset.py
"""

from huggingface_hub import HfApi, create_repo

HF_USERNAME = "djakobh"
DATASET_REPO = f"{HF_USERNAME}/neuroscan-testing"
LOCAL_DIR = "data/Testing"

api = HfApi()

print(f"Creating dataset repo: {DATASET_REPO}")
create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)

print("Uploading images (this will take a few minutes)...")
api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=DATASET_REPO,
    repo_type="dataset",
    commit_message="Upload Testing images",
)

print("Done. Dataset live at:")
print(f"  https://huggingface.co/datasets/{DATASET_REPO}")
