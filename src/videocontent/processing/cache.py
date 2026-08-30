"""Stage-level caching for incremental processing.

The cache key is: sha256(video_content_hash + stage_name + stage_version + config_hash)

This ensures that:
- Changing the video content invalidates all stages
- Changing a stage's config invalidates only that stage and its dependents
- Updating stage_version (when output format changes) invalidates the stage
- The cache is stored in the workspace's cache directory
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..logging import get_logger

log = get_logger("cache")


@dataclass
class CacheEntry:
    """A cached stage result."""
    stage_name: str
    stage_version: str
    video_hash: str
    config_hash: str
    created_at: float
    data: Any
    metadata: dict[str, Any]


class StageCache:
    """File-based cache for pipeline stages."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, video_hash: str, stage_name: str, stage_version: str, config_hash: str) -> str:
        """Generate a deterministic cache key."""
        key_string = f"{video_hash}|{stage_name}|{stage_version}|{config_hash}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.cache"

    def _meta_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.meta"

    def get(
        self,
        video_hash: str,
        stage_name: str,
        stage_version: str,
        config_hash: str,
    ) -> Optional[CacheEntry]:
        """Retrieve a cached stage result if valid."""
        if not video_hash:
            return None

        key = self._cache_key(video_hash, stage_name, stage_version, config_hash)
        cache_file = self._cache_path(key)
        meta_file = self._meta_path(key)

        if not cache_file.exists() or not meta_file.exists():
            return None

        try:
            with meta_file.open("r") as f:
                meta = json.load(f)

            # Verify the cache entry matches
            if (
                meta.get("stage_name") != stage_name or
                meta.get("stage_version") != stage_version or
                meta.get("video_hash") != video_hash or
                meta.get("config_hash") != config_hash
            ):
                log.debug("cache.invalidated", extra={"stage": stage_name, "reason": "metadata mismatch"})
                return None

            with cache_file.open("rb") as f:
                data = pickle.load(f)

            log.info("cache.hit", extra={"stage": stage_name, "key": key[:8]})
            return CacheEntry(
                stage_name=stage_name,
                stage_version=stage_version,
                video_hash=video_hash,
                config_hash=config_hash,
                created_at=meta.get("created_at", 0),
                data=data,
                metadata=meta.get("metadata", {}),
            )
        except Exception as exc:
            log.warning("cache.read_failed", extra={"stage": stage_name, "error": str(exc)})
            return None

    def set(
        self,
        video_hash: str,
        stage_name: str,
        stage_version: str,
        config_hash: str,
        data: Any,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Store a stage result in the cache."""
        if not video_hash:
            return

        key = self._cache_key(video_hash, stage_name, stage_version, config_hash)
        cache_file = self._cache_path(key)
        meta_file = self._meta_path(key)

        try:
            with cache_file.open("wb") as f:
                pickle.dump(data, f)

            meta = {
                "stage_name": stage_name,
                "stage_version": stage_version,
                "video_hash": video_hash,
                "config_hash": config_hash,
                "created_at": time.time(),
                "metadata": metadata or {},
            }
            with meta_file.open("w") as f:
                json.dump(meta, f)

            log.info("cache.stored", extra={"stage": stage_name, "key": key[:8]})
        except Exception as exc:
            log.warning("cache.write_failed", extra={"stage": stage_name, "error": str(exc)})

    def invalidate(self, video_hash: str, stage_name: Optional[str] = None) -> int:
        """Invalidate cache entries for a video (and optionally a specific stage)."""
        if not video_hash:
            return 0

        count = 0
        prefix = f"{video_hash}|{stage_name}|" if stage_name else f"{video_hash}|"
        for meta_file in self.cache_dir.glob("*.meta"):
            try:
                with meta_file.open("r") as f:
                    meta = json.load(f)
                if meta.get("video_hash") == video_hash and (stage_name is None or meta.get("stage_name") == stage_name):
                    key = meta_file.stem
                    cache_file = self.cache_dir / f"{key}.cache"
                    cache_file.unlink(missing_ok=True)
                    meta_file.unlink(missing_ok=True)
                    count += 1
            except Exception:
                continue

        if count > 0:
            log.info("cache.invalidated", extra={"count": count, "video_hash": video_hash[:12], "stage": stage_name})
        return count

    def clear(self) -> None:
        """Clear all cache entries."""
        for f in self.cache_dir.glob("*.cache"):
            f.unlink(missing_ok=True)
        for f in self.cache_dir.glob("*.meta"):
            f.unlink(missing_ok=True)
        log.info("cache.cleared")


__all__ = ["CacheEntry", "StageCache"]