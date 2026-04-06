from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any

from tex.domain.decision import Decision
from tex.domain.evidence import EvidenceRecord
from tex.domain.outcome import OutcomeRecord


class EvidenceRecorder:
    """
    Append-only JSONL evidence recorder with a tamper-evident hash chain.

    This recorder is deliberately small and strict:
    - writes canonical JSON payloads into an append-only log
    - wraps each payload in an EvidenceRecord envelope
    - maintains record-to-record linkage via previous_hash
    - does not own chain verification logic

    The domain contract for EvidenceRecord is the source of truth. This class
    must serialize into that contract exactly and must not invent parallel field
    names such as `payload` or `payload_hash`.
    """

    __slots__ = ("_path", "_lock", "_last_record_hash")

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._last_record_hash = self._load_last_record_hash()

    @property
    def path(self) -> Path:
        """Returns the backing JSONL path."""
        return self._path

    def record_decision(
        self,
        decision: Decision,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceRecord:
        """
        Appends an evidence record for a decision.

        The payload is intentionally verbose and audit-friendly. It captures the
        durable decision record plus optional command-layer metadata.
        """
        payload: dict[str, Any] = {
            "record_type": "decision",
            "decision_id": str(decision.decision_id),
            "request_id": str(decision.request_id),
            "verdict": decision.verdict.value,
            "confidence": decision.confidence,
            "final_score": decision.final_score,
            "action_type": decision.action_type,
            "channel": decision.channel,
            "environment": decision.environment,
            "recipient": decision.recipient,
            "policy_id": decision.policy_id,
            "policy_version": decision.policy_version,
            "content_excerpt": decision.content_excerpt,
            "content_sha256": decision.content_sha256,
            "scores": dict(decision.scores),
            "reasons": list(decision.reasons),
            "uncertainty_flags": list(decision.uncertainty_flags),
            "findings": [
                self._serialize_model(finding)
                for finding in decision.findings
            ],
            "retrieval_context": self._make_json_safe(decision.retrieval_context),
            "metadata": self._merge_metadata(decision.metadata, metadata),
            "evidence_hash": decision.evidence_hash,
            "decided_at": decision.decided_at.isoformat(),
        }

        return self._append(
            decision_id=decision.decision_id,
            request_id=decision.request_id,
            record_type="decision",
            policy_version=decision.policy_version,
            payload=payload,
        )

    def record_outcome(
        self,
        outcome: OutcomeRecord,
        *,
        metadata: dict[str, Any] | None = None,
        policy_version: str | None = None,
    ) -> EvidenceRecord:
        """
        Appends an evidence record for an outcome.

        OutcomeRecord does not carry policy_version directly, so callers may
        pass it explicitly. If omitted, this method will also accept
        `decision_policy_version` inside metadata for compatibility with the
        current command layer.
        """
        resolved_policy_version = self._resolve_outcome_policy_version(
            metadata=metadata,
            policy_version=policy_version,
        )

        payload: dict[str, Any] = {
            "record_type": "outcome",
            "outcome_id": str(outcome.outcome_id),
            "decision_id": str(outcome.decision_id),
            "request_id": str(outcome.request_id),
            "verdict": outcome.verdict.value,
            "outcome_kind": outcome.outcome_kind.value,
            "was_safe": outcome.was_safe,
            "human_override": outcome.human_override,
            "summary": outcome.summary,
            "reporter": outcome.reporter,
            "label": outcome.label.value,
            "policy_version": resolved_policy_version,
            "metadata": self._merge_metadata(None, metadata),
            "recorded_at": outcome.recorded_at.isoformat(),
        }

        return self._append(
            decision_id=outcome.decision_id,
            request_id=outcome.request_id,
            record_type="outcome",
            policy_version=resolved_policy_version,
            payload=payload,
        )

    def read_all(self) -> tuple[EvidenceRecord, ...]:
        """Reads and validates all evidence records from disk."""
        if not self._path.exists():
            return tuple()

        records: list[EvidenceRecord] = []

        with self._lock:
            with self._path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid JSON in evidence file at line {line_number}"
                        ) from exc

                    try:
                        record = EvidenceRecord.model_validate(payload)
                    except Exception as exc:
                        raise ValueError(
                            f"invalid evidence record at line {line_number}"
                        ) from exc

                    records.append(record)

        return tuple(records)

    def last_record(self) -> EvidenceRecord | None:
        """Returns the most recent evidence record, if any."""
        records = self.read_all()
        return records[-1] if records else None

    def decode_payload(self, record: EvidenceRecord) -> dict[str, Any]:
        """
        Parses the canonical payload_json for a stored evidence record.

        This is a convenience for higher layers such as exporters and filters.
        """
        try:
            value = json.loads(record.payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"evidence record {record.evidence_id} contains invalid payload_json"
            ) from exc

        if not isinstance(value, dict):
            raise ValueError(
                f"evidence record {record.evidence_id} payload_json must decode to an object"
            )

        return value

    def _append(
        self,
        *,
        decision_id: Any,
        request_id: Any,
        record_type: str,
        policy_version: str,
        payload: dict[str, Any],
    ) -> EvidenceRecord:
        with self._lock:
            payload_json = self._stable_json(self._make_json_safe(payload))
            payload_sha256 = self._sha256_hex(payload_json)

            record_hash = self._build_record_hash(
                payload_sha256=payload_sha256,
                previous_hash=self._last_record_hash,
            )

            record = EvidenceRecord(
                decision_id=decision_id,
                request_id=request_id,
                record_type=record_type,
                payload_json=payload_json,
                payload_sha256=payload_sha256,
                previous_hash=self._last_record_hash,
                record_hash=record_hash,
                policy_version=policy_version,
            )

            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(self._stable_json(record.model_dump(mode="json")))
                handle.write("\n")

            self._last_record_hash = record.record_hash
            return record

    def _load_last_record_hash(self) -> str | None:
        if not self._path.exists():
            return None

        last_non_empty_line: str | None = None

        with self._lock:
            with self._path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if line:
                        last_non_empty_line = line

        if last_non_empty_line is None:
            return None

        try:
            parsed = json.loads(last_non_empty_line)
            record = EvidenceRecord.model_validate(parsed)
        except Exception as exc:
            raise ValueError("failed to read last evidence record from file") from exc

        return record.record_hash

    @staticmethod
    def _resolve_outcome_policy_version(
        *,
        metadata: dict[str, Any] | None,
        policy_version: str | None,
    ) -> str:
        if policy_version is not None:
            normalized = policy_version.strip()
            if not normalized:
                raise ValueError("policy_version must not be blank")
            return normalized

        if metadata is not None:
            value = metadata.get("decision_policy_version")
            if isinstance(value, str):
                normalized = value.strip()
                if normalized:
                    return normalized

        raise ValueError(
            "record_outcome requires a policy_version or metadata['decision_policy_version']"
        )

    @staticmethod
    def _merge_metadata(
        base: dict[str, Any] | None,
        override: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}

        if base is not None:
            merged.update(dict(base))

        if override is not None:
            merged.update(dict(override))

        return EvidenceRecorder._make_json_safe(merged)

    @staticmethod
    def _serialize_model(value: Any) -> dict[str, Any]:
        if not hasattr(value, "model_dump"):
            raise TypeError("value must be a pydantic model with model_dump()")
        dumped = value.model_dump(mode="json")
        if not isinstance(dumped, dict):
            raise TypeError("serialized model must produce a JSON object")
        return dumped

    @staticmethod
    def _build_record_hash(
        *,
        payload_sha256: str,
        previous_hash: str | None,
    ) -> str:
        chain_input = EvidenceRecorder._stable_json(
            {
                "payload_sha256": payload_sha256,
                "previous_hash": previous_hash,
            }
        )
        return EvidenceRecorder._sha256_hex(chain_input)

    @staticmethod
    def _make_json_safe(value: Any) -> Any:
        """
        Normalizes arbitrary nested values into JSON-safe data.

        This keeps evidence serialization explicit and stable even when metadata
        contains UUIDs, datetimes, enums, tuples, or nested pydantic models.
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Path):
            return str(value)

        if hasattr(value, "isoformat") and callable(value.isoformat):
            try:
                return value.isoformat()
            except TypeError:
                pass

        if hasattr(value, "value"):
            enum_value = getattr(value, "value")
            if isinstance(enum_value, (str, int, float, bool)):
                return enum_value

        if hasattr(value, "model_dump") and callable(value.model_dump):
            return EvidenceRecorder._make_json_safe(value.model_dump(mode="json"))

        if isinstance(value, dict):
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                normalized[str(key)] = EvidenceRecorder._make_json_safe(item)
            return normalized

        if isinstance(value, (list, tuple, set, frozenset)):
            return [EvidenceRecorder._make_json_safe(item) for item in value]

        return str(value)

    @staticmethod
    def _stable_json(value: Any) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @staticmethod
    def _sha256_hex(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()