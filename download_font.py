"""
download_font.py

Downloads DejaVuSans.ttf from the official DejaVu project into ./fonts/.

Usage:
    python download_font.py
"""

import tarfile
import urllib.request
from io import BytesIO
from pathlib import Path

DEJAVU_URL = (
    "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/"
    "version_2_37/dejavu-fonts-ttf-2.37.tar.bz2"
)
TARGET_FILE_IN_ARCHIVE = "dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf"
OUTPUT_PATH = Path("./fonts/DejaVuSans.ttf")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"[INFO] Font already exists at {OUTPUT_PATH} — nothing to do.")
        return

    print(f"[INFO] Downloading DejaVu fonts archive from:\n  {DEJAVU_URL}")
    with urllib.request.urlopen(DEJAVU_URL) as resp:
        data = resp.read()

    print("[INFO] Extracting DejaVuSans.ttf...")
    with tarfile.open(fileobj=BytesIO(data), mode="r:bz2") as tf:
        member = tf.getmember(TARGET_FILE_IN_ARCHIVE)
        f = tf.extractfile(member)
        OUTPUT_PATH.write_bytes(f.read())

    print(f"[SUCCESS] Font saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
