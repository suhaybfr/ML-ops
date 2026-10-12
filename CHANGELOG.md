# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Tracked ASSISTments raw dataset (`skill_builder_data.csv`) with DVC under `data/raw/` via `data/raw/skill_builder_data.csv.dvc`.
- Added `data/raw/.gitignore` to exclude the raw dataset from Git tracking.
- Initialized DVC (Data Version Control) repository configuration (`.dvc/`, `.dvcignore`, `.dvc/.gitignore`).

### Changed
- Relocated original dataset `skill_builder_data.csv` from repository root to `data/raw/skill_builder_data.csv`, verifying MD5 checksum preservation (`9410c31fadc7ad6518d296ce254ec4ab`).
- Removed `skill_builder_data.csv` from Git tracking in the root folder.
- Removed obsolete `data/raw/placeholder.txt`.
