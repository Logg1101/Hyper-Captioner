from pathlib import Path
from datetime import datetime
import shutil


def create_backup(dataset_path):
    dataset_path = Path(dataset_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_folder = (
        dataset_path.parent
        / f"{dataset_path.name}_BACKUP_{timestamp}"
    )

    print("\nCreating backup...")
    print(f"Source : {dataset_path}")
    print(f"Backup : {backup_folder}")

    shutil.copytree(dataset_path, backup_folder)

    print("Backup completed.")

    return backup_folder