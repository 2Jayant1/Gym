# Kaggle API Setup

1) Create API token
- Go to https://www.kaggle.com/settings > API > "Create New Token" (downloads `kaggle.json`).

2) Place the token
- Windows: `C:\Users\<USERNAME>\.kaggle\kaggle.json`
- Mac/Linux: `~/.kaggle/kaggle.json`

3) Permissions (Mac/Linux)
- `chmod 600 ~/.kaggle/kaggle.json`

4) Install CLI dependencies
- `python -m pip install -r requirements.txt`

5) Verify
- `python -m kaggle --version`

6) Download
- `python scripts/download_dataset.py`
