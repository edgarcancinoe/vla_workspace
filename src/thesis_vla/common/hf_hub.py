from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi


FAILED_HUB_UPLOAD_FILENAME = "hf_upload_failed.json"


@dataclass(frozen=True)
class HubUploadConfig:
    max_retries: int = 5
    retry_backoff_s: float = 5.0


@dataclass(frozen=True)
class HubUploadResult:
    ok: bool
    attempts: int
    error: str | None = None


def failed_hub_upload_marker_path(folder_path: str | Path) -> Path:
    return Path(folder_path) / FAILED_HUB_UPLOAD_FILENAME


def clear_hub_upload_failure_marker(folder_path: str | Path) -> None:
    failed_hub_upload_marker_path(folder_path).unlink(missing_ok=True)


def write_hub_upload_failure_marker(folder_path: str | Path, repo_id: str, repo_type: str, commit_message: str, result: HubUploadResult, path_in_repo: str | None = None) -> Path:
    marker_path = failed_hub_upload_marker_path(folder_path)
    payload = {"repo_id": repo_id, "repo_type": repo_type, "folder_path": str(Path(folder_path)), "path_in_repo": path_in_repo, "commit_message": commit_message, "attempts": int(result.attempts), "error": result.error, "failed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return marker_path


def push_folder_to_hub(folder_path: str | Path, repo_id: str, repo_type: str, commit_message: str, ignore_patterns: list[str] | None = None, delete_patterns: list[str] | None = None, path_in_repo: str | None = None, upload_config: HubUploadConfig | None = None, logger=print) -> HubUploadResult:
    upload_config = upload_config or HubUploadConfig()
    max_retries = max(int(upload_config.max_retries), 1)
    api = HfApi()
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger(f"[push] Upload attempt {attempt}/{max_retries} -> {repo_id}")
            api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
            api.upload_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type=repo_type, ignore_patterns=ignore_patterns, delete_patterns=delete_patterns, path_in_repo=path_in_repo, commit_message=commit_message)
            logger(f"[push] Successfully pushed {repo_type} repo {repo_id}")
            return HubUploadResult(ok=True, attempts=attempt)
        except Exception as error:
            last_error = error
            logger(f"[push] Attempt {attempt} failed for {repo_id}: {error}")
            if attempt >= max_retries:
                break
            sleep_s = max(float(upload_config.retry_backoff_s), 0.0) * attempt
            if sleep_s > 0:
                logger(f"[push] Retrying in {sleep_s:.1f}s...")
                time.sleep(sleep_s)
    message = f"{type(last_error).__name__}: {last_error}" if last_error is not None else "unknown error"
    return HubUploadResult(ok=False, attempts=max_retries, error=message)
