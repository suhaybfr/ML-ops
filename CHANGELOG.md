# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Configured DAGsHub as default DVC remote storage (`origin`) at `https://dagshub.com/suhaybfr/ML-ops.dvc`.
- Uploaded raw ASSISTments dataset to DAGsHub remote storage via `dvc push`.
- Stored remote authentication securely in `.dvc/config.local` (excluded from Git).
- Added DVC documentation and DAGsHub remote workflow instructions to `README.md`.
- Tracked ASSISTments raw dataset (`skill_builder_data.csv`) with DVC under `data/raw/` via `data/raw/skill_builder_data.csv.dvc`.
- Added `data/raw/.gitignore` to exclude the raw dataset from Git tracking.
- Initialized DVC (Data Version Control) repository configuration (`.dvc/`, `.dvcignore`, `.dvc/.gitignore`).

### Changed
- Updated `README.md` project structure diagram, remote storage instructions, and current project status.
- Relocated original dataset `skill_builder_data.csv` from repository root to `data/raw/skill_builder_data.csv`, verifying MD5 checksum preservation (`9410c31fadc7ad6518d296ce254ec4ab`).
- Removed `skill_builder_data.csv` from Git tracking in the root folder.
- Removed obsolete `data/raw/placeholder.txt`.
