"""Versioned model artifact storage backed by Azure Blob Storage (with local fallback).

When ``AZURE_STORAGE_CONNECTION_STRING`` is configured, uploads versioned
challenger / promoted ``.pkl`` artifacts to the ``AZURE_STORAGE_CONTAINER``
container (default ``models``) so out-of-band retraining jobs (Azure Container
Apps Jobs / GitHub Actions) and serving containers share immutable artifacts.
When unset, stores versioned copies cleanly under ``models/artifacts/`` for
100% offline CI safety.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_ARTIFACTS_DIR = REPO_ROOT / "models" / "artifacts"


def is_azure_blob_configured() -> bool:
    """Return True when ``AZURE_STORAGE_CONNECTION_STRING`` is set."""
    return bool(os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip())


class ModelArtifactStore:
    """Upload, download, and list versioned ``.pkl`` artifacts in Azure Blob or locally."""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        container_name: Optional[str] = None,
        local_dir: Optional[Path] = None,
    ) -> None:
        self.connection_string = (
            connection_string
            if connection_string is not None
            else os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
        )
        self.container_name = (
            container_name
            or os.environ.get("AZURE_STORAGE_CONTAINER", "models").strip()
            or "models"
        )
        self.local_dir = local_dir or LOCAL_ARTIFACTS_DIR

    @property
    def backend_name(self) -> str:
        return "azure_blob" if self.connection_string else "local"

    def _get_container_client(self):
        from azure.storage.blob import BlobServiceClient

        svc = BlobServiceClient.from_connection_string(self.connection_string)
        client = svc.get_container_client(self.container_name)
        try:
            client.create_container()
        except Exception:
            # Container already exists (409 ResourceExistsError) or managed externally
            pass
        return client

    def upload_model(
        self,
        local_pkl_path: str | Path,
        blob_name: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Upload a ``.pkl`` artifact and return its storage descriptor."""
        src = Path(local_pkl_path)
        if not src.exists():
            raise FileNotFoundError(f"Model artifact not found: {src}")

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target_name = blob_name or f"{src.stem}_{stamp}.pkl"
        clean_meta = {str(k): str(v) for k, v in (metadata or {}).items()}

        if self.connection_string:
            container = self._get_container_client()
            blob_client = container.get_blob_client(target_name)
            with open(src, "rb") as fh:
                blob_client.upload_blob(fh, overwrite=True, metadata=clean_meta or None)
            return {
                "backend": "azure_blob",
                "container": self.container_name,
                "blob_name": target_name,
                "uri": f"az://{self.container_name}/{target_name}",
                "size_bytes": src.stat().st_size,
            }

        self.local_dir.mkdir(parents=True, exist_ok=True)
        dest = self.local_dir / Path(target_name).name
        shutil.copy2(src, dest)
        return {
            "backend": "local",
            "container": str(self.local_dir),
            "blob_name": dest.name,
            "uri": str(dest),
            "size_bytes": dest.stat().st_size,
        }

    def download_model(self, blob_name: str, dest_path: str | Path) -> Path:
        """Download a versioned ``.pkl`` artifact from Azure Blob or local store."""
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if self.connection_string:
            clean_name = blob_name
            prefix = f"az://{self.container_name}/"
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
            container = self._get_container_client()
            blob_client = container.get_blob_client(clean_name)
            with open(dest, "wb") as fh:
                fh.write(blob_client.download_blob().readall())
            return dest

        src = Path(blob_name)
        if not src.is_absolute():
            candidate = self.local_dir / blob_name
            src = candidate if candidate.exists() else src
        if not src.exists():
            raise FileNotFoundError(f"Local artifact not found: {blob_name}")
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return dest

    def list_models(self) -> list[str]:
        """List stored versioned ``.pkl`` artifact names."""
        if self.connection_string:
            container = self._get_container_client()
            return sorted(
                b.name for b in container.list_blobs() if b.name.endswith(".pkl")
            )
        if not self.local_dir.exists():
            return []
        return sorted(p.name for p in self.local_dir.glob("*.pkl"))

    def delete_non_baseline_models(self) -> list[str]:
        """Delete demo/retrained ``.pkl`` artifacts while strictly preserving baseline_model.pkl."""
        deleted: list[str] = []
        if self.connection_string:
            container = self._get_container_client()
            for blob in container.list_blobs():
                if blob.name.endswith(".pkl") and blob.name != "baseline_model.pkl":
                    container.delete_blob(blob.name)
                    deleted.append(blob.name)
            return deleted
        if self.local_dir.exists():
            for p in self.local_dir.glob("*.pkl"):
                if p.name != "baseline_model.pkl":
                    p.unlink()
                    deleted.append(p.name)
        return deleted
