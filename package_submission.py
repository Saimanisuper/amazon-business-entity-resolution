#!/usr/bin/env python3
"""
ML Challenge 2026 — Submission Archive Builder

Generates the exact submission zip file required by the competition:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path


def create_submission_zip(team_name: str = "Mighty", output_zip_path: Path = None):
    base_dir = Path(__file__).resolve().parent
    if output_zip_path is None:
        output_zip_path = base_dir / f"{team_name}_submission.zip"

    # Required files to check
    required_files = [
        base_dir / "output" / "matching_results.tsv",
        base_dir / "output" / "candidate_pairs.tsv",
        base_dir / "code" / "business_entity_resolution" / "README.md",
        base_dir / "code" / "business_entity_resolution" / "requirements.txt",
        base_dir / "Documentation_template.md",
    ]

    missing = [str(f.relative_to(base_dir)) for f in required_files if not f.is_file()]
    if missing:
        print("ERROR: Missing required submission files:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)

    src_dir = base_dir / "code" / "business_entity_resolution" / "src"
    if not src_dir.is_dir():
        print(f"ERROR: Missing source directory: {src_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Building submission zip: {output_zip_path.name}")
    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        zf.write(base_dir / "output" / "matching_results.tsv", "output/matching_results.tsv")
        zf.write(base_dir / "output" / "candidate_pairs.tsv", "output/candidate_pairs.tsv")

        # 2. Code docs & requirements
        zf.write(
            base_dir / "code" / "business_entity_resolution" / "README.md",
            "code/business_entity_resolution/README.md"
        )
        zf.write(
            base_dir / "code" / "business_entity_resolution" / "requirements.txt",
            "code/business_entity_resolution/requirements.txt"
        )

        # 3. Code source directory
        for root, dirs, files in os.walk(src_dir):
            if "__pycache__" in root:
                continue
            for file in files:
                if file.endswith((".pyc", ".pyo")) or file.startswith("."):
                    continue
                file_path = Path(root) / file
                arcname = Path("code") / "business_entity_resolution" / "src" / file_path.relative_to(src_dir)
                zf.write(file_path, str(arcname).replace("\\", "/"))

        # 4. Methodology Document
        zf.write(base_dir / "Documentation_template.md", "Documentation_template.md")

    print("\n[SUCCESS] Archive created successfully.")
    print("Archive Contents:")
    with zipfile.ZipFile(output_zip_path, "r") as zf:
        for info in zf.infolist():
            size_kb = info.file_size / 1024
            print(f"  {info.filename:<50} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Amazon ML Challenge 2026 Submission Zip")
    parser.add_argument("--team-name", type=str, default="Mighty", help="Team name prefix for zip archive")
    parser.add_argument("--output", type=str, default=None, help="Custom output zip path")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else None
    create_submission_zip(team_name=args.team_name, output_zip_path=out_path)
