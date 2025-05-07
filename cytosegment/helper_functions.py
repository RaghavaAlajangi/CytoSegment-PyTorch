import hashlib
from pathlib import Path


def compute_md5(file_path, characters=5):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            md5_hash.update(chunk)
    md5_checksum = md5_hash.hexdigest()
    return md5_checksum[:characters]


def rename_ckp_path_with_md5(file_path, ITER=1):
    org_path = Path(file_path)
    md5_checksum = compute_md5(file_path)
    # Create file path with md5 sum
    new_file_name = f"{org_path.stem}_g{ITER}_{md5_checksum}{org_path.suffix}"
    new_file_path = org_path.parent / new_file_name
    # Rename original path with new path
    org_path.rename(new_file_path)
    return new_file_path
