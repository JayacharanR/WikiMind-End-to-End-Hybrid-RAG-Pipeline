"""Pipeline Health Tracker & Dead Letter Queue.

Shared health/metrics module for WikiMind data pipeline workers
(wiki_updater, reconciler). Provides:

- PipelineHealthTracker: tracks events processed, failed, uptime, heartbeat
- Dead Letter Queue (DLQ): stores failed events with error metadata
- DLQ retry logic: periodically re-processes failed events
- JSON file persistence: DLQ survives container/process restarts
- Structured alerting: severity-tagged log messages

Usage::

    from data_pipeline.pipeline_health import PipelineHealthTracker

    tracker = PipelineHealthTracker(name="wiki-updater")
    tracker.record_success(event_id="Paris")
    tracker.add_to_dlq(event={"title": "Paris"}, error="timeout")
    tracker.retry_dlq(process_fn=my_handler)
"""

import json
import logging
import os
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Default DLQ persistence path
_DLQ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_MAX_DLQ_SIZE = 1000
_MAX_DLQ_RETRIES = 3


class PipelineHealthTracker:
    """Tracks health, metrics, and DLQ for a pipeline worker.

    Attributes:
        name: Human-readable worker name (e.g., 'wiki-updater', 'reconciler').
        events_processed: Total successfully processed events since startup.
        events_failed: Total failed events since startup.
        last_heartbeat: ISO timestamp of last successful event processing.
        started_at: ISO timestamp when the tracker was initialized.
        dlq: In-memory deque of failed events with error metadata.
        consecutive_failures: Count of failures in a row (resets on success).
    """

    def __init__(self, name: str, dlq_persist_path: Optional[str] = None):
        self.name = name
        self.events_processed = 0
        self.events_failed = 0
        self.consecutive_failures = 0
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.last_heartbeat: Optional[str] = None
        self.last_error: Optional[str] = None
        self.dlq: deque = deque(maxlen=_MAX_DLQ_SIZE)
        self._dlq_lock = threading.RLock()

        # DLQ persistence
        if dlq_persist_path:
            self._dlq_path = dlq_persist_path
        else:
            os.makedirs(_DLQ_DIR, exist_ok=True)
            self._dlq_path = os.path.join(_DLQ_DIR, f"dlq_{name}.json")

        # Load persisted DLQ on startup
        self._load_dlq()

        # Drift/reconciliation-specific metrics
        self.drift_detected = 0
        self.reingestion_success = 0
        self.reingestion_failed = 0
        self.last_cycle_stats: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Event Tracking
    # ------------------------------------------------------------------

    def record_success(self, event_id: str = "") -> None:
        """Record a successfully processed event."""
        self.events_processed += 1
        self.consecutive_failures = 0
        self.last_heartbeat = datetime.now(timezone.utc).isoformat()

    def record_failure(self, event_id: str = "", error: str = "") -> None:
        """Record a failed event (without adding to DLQ)."""
        self.events_failed += 1
        self.consecutive_failures += 1
        self.last_error = error

        # Alert on consecutive failures
        if self.consecutive_failures >= 5:
            self.alert(
                "WARNING",
                f"[{self.name}] {self.consecutive_failures} consecutive failures. Last error: {error}",
            )
        if self.consecutive_failures >= 20:
            self.alert(
                "CRITICAL",
                f"[{self.name}] {self.consecutive_failures} consecutive failures — possible systemic issue",
            )

    # ------------------------------------------------------------------
    # Dead Letter Queue
    # ------------------------------------------------------------------

    def add_to_dlq(self, event: Dict[str, Any], error_msg: str) -> None:
        """Add a failed event to the Dead Letter Queue with metadata.

        Args:
            event: The original event data that failed processing.
            error_msg: Human-readable error description.
        """
        dlq_entry = {
            "event": event,
            "error": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "retries": 0,
            "worker": self.name,
        }
        self.dlq.append(dlq_entry)
        self.record_failure(
            event_id=event.get("title", "unknown"),
            error=error_msg,
        )
        logger.warning(
            "[%s] Event added to DLQ (size=%d): %s — %s",
            self.name,
            len(self.dlq),
            event.get("title", "unknown")[:60],
            error_msg[:100],
        )

        # Persist to disk after every addition
        self._save_dlq()

    async def retry_dlq(
        self,
        process_fn: Callable,
        max_retries: int = _MAX_DLQ_RETRIES,
        session: Any = None,
    ) -> Dict[str, int]:
        """Attempt to re-process DLQ items.

        Args:
            process_fn: Async callable that processes an event dict.
                Signature: process_fn(event_data, session) -> None
            max_retries: Maximum retry attempts per DLQ entry.
            session: Optional aiohttp session to pass to process_fn.

        Returns:
            Dict with 'retried', 'succeeded', 'permanently_failed' counts.
        """
        if not self.dlq:
            return {"retried": 0, "succeeded": 0, "permanently_failed": 0}

        stats = {"retried": 0, "succeeded": 0, "permanently_failed": 0}
        remaining = deque(maxlen=_MAX_DLQ_SIZE)

        while self.dlq:
            entry = self.dlq.popleft()
            entry["retries"] += 1
            stats["retried"] += 1

            try:
                if session is not None:
                    await process_fn(entry["event"], session)
                else:
                    await process_fn(entry["event"])
                stats["succeeded"] += 1
                self.events_processed += 1
                logger.info(
                    "[%s] DLQ retry succeeded: %s (attempt %d)",
                    self.name,
                    entry["event"].get("title", "unknown"),
                    entry["retries"],
                )
            except Exception as exc:
                if entry["retries"] >= max_retries:
                    stats["permanently_failed"] += 1
                    self.alert(
                        "ERROR",
                        f"[{self.name}] DLQ item permanently failed after {max_retries} retries: "
                        f"{entry['event'].get('title', 'unknown')} — {exc}",
                    )
                else:
                    entry["error"] = str(exc)
                    entry["last_retry"] = datetime.now(timezone.utc).isoformat()
                    remaining.append(entry)

        self.dlq = remaining
        self._save_dlq()

        if stats["retried"] > 0:
            logger.info(
                "[%s] DLQ retry summary: %d retried, %d succeeded, %d permanently failed, %d remaining",
                self.name,
                stats["retried"],
                stats["succeeded"],
                stats["permanently_failed"],
                len(self.dlq),
            )

        return stats

    # ------------------------------------------------------------------
    # Reconciler-Specific Metrics
    # ------------------------------------------------------------------

    def record_cycle_stats(
        self,
        drift_count: int,
        success_count: int,
        failed_count: int,
        elapsed_sec: float,
    ) -> None:
        """Record metrics from a single reconciliation cycle."""
        self.drift_detected += drift_count
        self.reingestion_success += success_count
        self.reingestion_failed += failed_count
        self.last_cycle_stats = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "drift_detected": drift_count,
            "reingestion_success": success_count,
            "reingestion_failed": failed_count,
            "elapsed_sec": round(elapsed_sec, 1),
        }
        self.last_heartbeat = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Health Status
    # ------------------------------------------------------------------

    def get_health_status(self) -> Dict[str, Any]:
        """Return a structured health report for API consumption."""
        uptime_sec = 0.0
        if self.started_at:
            try:
                start = datetime.fromisoformat(self.started_at)
                uptime_sec = (datetime.now(timezone.utc) - start).total_seconds()
            except Exception:
                pass

        return {
            "worker": self.name,
            "status": "healthy" if self.consecutive_failures < 10 else "degraded",
            "events_processed": self.events_processed,
            "events_failed": self.events_failed,
            "consecutive_failures": self.consecutive_failures,
            "dlq_size": len(self.dlq),
            "last_heartbeat": self.last_heartbeat,
            "last_error": self.last_error,
            "started_at": self.started_at,
            "uptime_seconds": round(uptime_sec),
            # Reconciler-specific (None for updater)
            "drift_detected": self.drift_detected if self.drift_detected else None,
            "reingestion_success": self.reingestion_success if self.reingestion_success else None,
            "last_cycle": self.last_cycle_stats,
        }

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------

    @staticmethod
    def alert(level: str, message: str) -> None:
        """Emit a structured alert log.

        In a production environment, this would integrate with PagerDuty,
        Slack webhooks, or similar. For now, it produces a clearly-tagged
        log line that can be filtered by log aggregators.

        Args:
            level: One of 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
            message: Alert message body.
        """
        tag = f"[ALERT:{level}]"
        if level == "CRITICAL":
            logger.critical("%s %s", tag, message)
        elif level == "ERROR":
            logger.error("%s %s", tag, message)
        elif level == "WARNING":
            logger.warning("%s %s", tag, message)
        else:
            logger.info("%s %s", tag, message)

    # ------------------------------------------------------------------
    # DLQ Persistence
    # ------------------------------------------------------------------

    def _save_dlq(self) -> None:
        """Persist the DLQ to a JSON file on disk."""
        try:
            with self._dlq_lock:
                data = list(self.dlq)
                temp_path = f"{self._dlq_path}.tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                os.replace(temp_path, self._dlq_path)
        except Exception as exc:
            logger.debug("Failed to persist DLQ for %s: %s", self.name, exc)

    def _load_dlq(self) -> None:
        """Load persisted DLQ from disk on startup."""
        if not os.path.exists(self._dlq_path):
            return
        try:
            with open(self._dlq_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                for entry in data[-_MAX_DLQ_SIZE:]:
                    self.dlq.append(entry)
                if self.dlq:
                    logger.info(
                        "[%s] Loaded %d DLQ items from disk.",
                        self.name,
                        len(self.dlq),
                    )
        except Exception as exc:
            logger.warning("Failed to load persisted DLQ for %s: %s", self.name, exc)
