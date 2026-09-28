"""
download_model.py

Convenience script to convert Qwen2.5-VL-7B-Instruct from HuggingFace
to an OpenVINO INT8 model using optimum-intel.

Usage:
    python download_model.py [--device CPU]

Requirements:
    pip install optimum[openvino] nncf huggingface_hub
"""

import argparse
import subprocess
import sys
from pathlib import Path


HF_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR = Path("./models/Qwen2.5-VL-7B-Instruct-int8-ov")


def main():
    parser = argparse.ArgumentParser(description="Download & convert Qwen2.5-VL to OV INT8")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        default=HF_MODEL_ID,
        help=f"HuggingFace model ID (default: {HF_MODEL_ID})",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        print(f"[INFO] Output directory already exists and is non-empty: {output}")
        print("[INFO] Delete it or choose a different --output path to re-convert.")
        sys.exit(0)

    output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "optimum.exporters.openvino",
        "--model", args.model,
        "--weight-format", "int8",
        "--trust-remote-code",
        str(output),
    ]

    print(f"[INFO] Running: {' '.join(cmd)}")
    print("[INFO] This may take 10-30 minutes and requires ~32 GB RAM.")
    print("[INFO] The HuggingFace model (~15 GB) will be downloaded first.")
    print()

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("\n[ERROR] Conversion failed. Try running manually:")
        print(f"  optimum-cli export openvino \\")
        print(f"      --model {args.model} \\")
        print(f"      --weight-format int8 \\")
        print(f"      {output}")
        sys.exit(1)

    print(f"\n[SUCCESS] Model saved to: {output}")
    print("You can now start the service with:")
    print("  uvicorn main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
