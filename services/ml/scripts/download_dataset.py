import shutil
import subprocess
from pathlib import Path
import sys
import sysconfig
import zipfile

from dataset_config import DATASETS, RAW_DATA_DIR, PROCESSED_DATA_DIR


def log(msg: str) -> None:
    print(msg, flush=True)


def find_kaggle_cli() -> list[str]:
    """Locate kaggle CLI binary; fall back to python -m kaggle."""
    # 1) If on PATH
    found = shutil.which("kaggle")
    if found:
        return [found]

    # 2) Common script dirs (user and base)
    candidates = []
    scripts_path = sysconfig.get_path("scripts")
    if scripts_path:
        candidates.append(Path(scripts_path) / "kaggle.exe")
        candidates.append(Path(scripts_path) / "kaggle")

    # user base (site.USER_BASE/Scripts)
    try:
        import site

        user_base = getattr(site, "USER_BASE", None) or site.getuserbase()
        candidates.append(Path(user_base) / "Scripts" / "kaggle.exe")
        candidates.append(Path(user_base) / "Scripts" / "kaggle")
    except Exception:
        pass

    # 2b) Windows user install typical path (AppData/Roaming/Python/<ver>/Scripts)
    home = Path.home()
    py_ver = f"Python{sys.version_info.major}{sys.version_info.minor}"
    candidates.append(home / "AppData" / "Roaming" / "Python" / py_ver / "Scripts" / "kaggle.exe")
    candidates.append(home / "AppData" / "Roaming" / "Python" / py_ver / "Scripts" / "kaggle")

    for cand in candidates:
        if cand and cand.exists():
            return [str(cand)]

    # 3) Fallback to python -m kaggle (may fail if package missing __main__)
    return [sys.executable, "-m", "kaggle"]


def already_processed(processed_dir: Path) -> bool:
    return any(processed_dir.iterdir())


def ensure_dirs(raw_dir: Path, processed_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)


def download_zip(raw_dir: Path, dataset: str) -> Path:
    zip_path = raw_dir / f"{dataset.split('/')[-1]}.zip"
    if zip_path.exists():
        log(f"[INFO] Found existing download: {zip_path}")
        return zip_path

    cli = find_kaggle_cli()
    cmd = [
        *cli,
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(raw_dir),
    ]
    log("[INFO] Download started")
    subprocess.run(cmd, check=True)

    if zip_path.exists():
        return zip_path

    # Fallback: pick latest zip in raw_dir
    zips = sorted(raw_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise FileNotFoundError("No zip file found after download")
    return zips[0]


def extract_zip(zip_path: Path, processed_dir: Path) -> None:
    log("[INFO] Extracting files")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(processed_dir)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / RAW_DATA_DIR
    processed_dir = project_root / PROCESSED_DATA_DIR

    ensure_dirs(raw_dir, processed_dir)

    failed = []
    for dataset in DATASETS:
        slug = dataset.split("/")[-1]
        ds_processed = processed_dir / slug
        ds_processed.mkdir(parents=True, exist_ok=True)

        if any(ds_processed.iterdir()):
            log(f"[SKIP] {slug} already extracted.")
            continue

        try:
            zip_path = download_zip(raw_dir, dataset)
            extract_zip(zip_path, ds_processed)
            log(f"[SUCCESS] {slug} ready")
        except FileNotFoundError as e:
            log(f"[ERROR] {slug}: {e}")
            failed.append(slug)
        except subprocess.CalledProcessError as e:
            log(f"[ERROR] {slug}: Kaggle CLI exit code {e.returncode}")
            failed.append(slug)
        except zipfile.BadZipFile:
            log(f"[ERROR] {slug}: bad zip")
            failed.append(slug)

    if failed:
        log(f"[WARN] Failed datasets: {', '.join(failed)}")
        return 1

    log("[SUCCESS] All datasets ready in data/processed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
