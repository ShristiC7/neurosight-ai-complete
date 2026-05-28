"""
NeuroSight AI — Eye Fatigue Data Preparation Utility
Scans the dataset directories, parses MRL filename conventions,
and populates labels.csv with structured metadata.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_mrl_filename(filename: str) -> dict[str, str | int] | None:
    """
    Parses metadata from an MRL Eye Dataset filename.
    Format: s[SubjectID]_[ImageID]_[Gender]_[Glasses]_[EyeState]_[Reflections]_[Lighting]_[SensorID].png
    """
    # Remove file extension and split by underscores
    stem = Path(filename).stem
    parts = stem.split("_")
    
    if len(parts) < 8:
        logger.warning(f"Skipping file with unrecognized format: {filename}")
        return None
        
    try:
        return {
            "subject_id": parts[0],
            "image_id": parts[1],
            "gender": int(parts[2]),
            "glasses": int(parts[3]),
            "eye_state": int(parts[4]),
            "reflections": int(parts[5]),
            "lighting": int(parts[6]),
            "sensor_id": parts[7]
        }
    except ValueError as e:
        logger.warning(f"Error parsing metadata from filename '{filename}': {e}")
        return None


def generate_labels_for_dataset(dataset_dir: Path, subdirs: list[str]) -> None:
    """
    Scans subdirs under dataset_dir, parses filenames, and writes labels.csv.
    """
    csv_path = dataset_dir / "labels.csv"
    logger.info(f"Generating labels for dataset at {dataset_dir}...")

    headers = [
        "filename", "subject_id", "image_id", "gender", 
        "glasses", "eye_state", "reflections", "lighting", "sensor_id"
    ]
    
    rows = []
    total_scanned = 0
    total_parsed = 0

    for subdir in subdirs:
        subdir_path = dataset_dir / subdir
        if not subdir_path.exists():
            logger.warning(f"Subdirectory not found: {subdir_path}")
            continue
            
        logger.info(f"Scanning subdirectory '{subdir}'...")
        # Supported image extensions
        extensions = ("*.png", "*.jpg", "*.jpeg")
        for ext in extensions:
            for file_path in subdir_path.glob(ext):
                total_scanned += 1
                rel_path = f"{subdir}/{file_path.name}"
                metadata = parse_mrl_filename(file_path.name)
                
                if metadata:
                    row = {"filename": rel_path}
                    row.update(metadata)
                    rows.append(row)
                    total_parsed += 1

    if not rows:
        logger.error(f"No valid image files found in {dataset_dir}")
        return

    # Sort rows by subject_id and image_id for cleanliness
    rows.sort(key=lambda r: (r["subject_id"], r["image_id"]))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        
    logger.info(
        f"Successfully generated {csv_path.name} in {dataset_dir}. "
        f"Parsed {total_parsed}/{total_scanned} files."
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    
    # 1. Drowsiness Detection Dataset
    drowsiness_dir = base_dir / "drowsiness-detection"
    generate_labels_for_dataset(drowsiness_dir, ["closed_eye", "open_eye"])
    
    # 2. MRL Dataset (Train)
    mrl_train_dir = base_dir / "mrl-dataset" / "train"
    generate_labels_for_dataset(mrl_train_dir, ["Closed_Eyes", "Open_Eyes"])


if __name__ == "__main__":
    main()
