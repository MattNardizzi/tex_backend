from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from tex.domain.evidence import EvidenceRecord
from tex.evidence.chain import ChainVerificationResult, verify_evidence_chain
from tex.evidence.recorder import EvidenceRecorder


@dataclass(frozen=True, slots=True)
class EvidenceExportBundle:
    """
    Export-ready evidence bundle.

    This packages raw evidence envelopes together with chain-verification
    results and lightweight export metadata.
    """

    export_name: str
    record_count: int
    is_chain_valid: bool
    verification: ChainVerificationResult
    records: tuple[EvidenceRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """Returns a fully JSON-serializable representation of the bundle."""
        return {
            "export_name": self.export_name,
            "record_count": self.record_count,
            "is_chain_valid": self.is_chain_valid,
            "verification": {
                "is_valid": self.verification.is_valid,
                "record_count": self.verification.record_count,
                "issue_count": self.verification.issue_count,
                "issues": [
                    {
                        "index": issue.index,
                        "record_hash": issue.record_hash,
                        "code": issue.code,
                        "message": issue.message,
                    }
                    for issue in self.verification.issues
                ],
            },
            "records": [
                record.model_dump(mode="json")
                for record in self.records
            ],
        }


class EvidenceExporter:
    """
    Exports Tex evidence records into portable audit bundles.

    Responsibilities:
    - read evidence envelopes from the recorder
    - optionally verify chain integrity
    - export either wrapped JSON bundles or raw JSONL envelopes
    - support filtering by decoded payload content without mutating records
    """

    __slots__ = ("_recorder",)

    def __init__(self, recorder: EvidenceRecorder) -> None:
        self._recorder = recorder

    def build_bundle(
        self,
        *,
        export_name: str = "tex-evidence-bundle",
        verify_chain: bool = True,
    ) -> EvidenceExportBundle:
        """
        Builds an in-memory evidence bundle from all stored records.
        """
        records = self._recorder.read_all()
        verification = self._build_verification(records, verify_chain=verify_chain)

        return EvidenceExportBundle(
            export_name=export_name,
            record_count=len(records),
            is_chain_valid=verification.is_valid,
            verification=verification,
            records=records,
        )

    def export_json(
        self,
        path: str | Path,
        *,
        export_name: str = "tex-evidence-bundle",
        verify_chain: bool = True,
        indent: int = 2,
    ) -> Path:
        """
        Writes a JSON evidence bundle to disk and returns the output path.
        """
        bundle = self.build_bundle(
            export_name=export_name,
            verify_chain=verify_chain,
        )

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                bundle.to_dict(),
                handle,
                indent=indent,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")

        return output_path

    def export_jsonl(
        self,
        path: str | Path,
    ) -> Path:
        """
        Exports raw evidence envelopes as JSONL.

        This preserves the exact stored records and is the safest format for
        replay, transport, or independent verification.
        """
        records = self._recorder.read_all()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record.model_dump(mode="json"),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                )
                handle.write("\n")

        return output_path

    def export_filtered_json(
        self,
        path: str | Path,
        *,
        record_type: str | None = None,
        decision_id: str | UUID | None = None,
        outcome_id: str | UUID | None = None,
        request_id: str | UUID | None = None,
        policy_version: str | None = None,
        export_name: str = "tex-evidence-filtered-bundle",
        verify_chain: bool = False,
        indent: int = 2,
    ) -> Path:
        """
        Exports a filtered JSON bundle.

        Filtering is performed against the evidence payload and selected envelope
        fields. Chain verification is disabled by default because an arbitrary
        filtered subset is usually not a valid contiguous chain.
        """
        records = self._recorder.read_all()

        filtered = tuple(
            record
            for record in records
            if self._matches_filters(
                record,
                record_type=record_type,
                decision_id=decision_id,
                outcome_id=outcome_id,
                request_id=request_id,
                policy_version=policy_version,
            )
        )

        verification = self._build_verification(filtered, verify_chain=verify_chain)

        bundle = EvidenceExportBundle(
            export_name=export_name,
            record_count=len(filtered),
            is_chain_valid=verification.is_valid,
            verification=verification,
            records=filtered,
        )

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                bundle.to_dict(),
                handle,
                indent=indent,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")

        return output_path

    def filter_records(
        self,
        *,
        record_type: str | None = None,
        decision_id: str | UUID | None = None,
        outcome_id: str | UUID | None = None,
        request_id: str | UUID | None = None,
        policy_version: str | None = None,
    ) -> tuple[EvidenceRecord, ...]:
        """
        Returns filtered evidence records without exporting them.
        """
        records = self._recorder.read_all()
        return tuple(
            record
            for record in records
            if self._matches_filters(
                record,
                record_type=record_type,
                decision_id=decision_id,
                outcome_id=outcome_id,
                request_id=request_id,
                policy_version=policy_version,
            )
        )

    def _build_verification(
        self,
        records: tuple[EvidenceRecord, ...],
        *,
        verify_chain: bool,
    ) -> ChainVerificationResult:
        if verify_chain:
            return verify_evidence_chain(records)

        return ChainVerificationResult(
            is_valid=True,
            record_count=len(records),
            issues=tuple(),
        )

    def _matches_filters(
        self,
        record: EvidenceRecord,
        *,
        record_type: str | None,
        decision_id: str | UUID | None,
        outcome_id: str | UUID | None,
        request_id: str | UUID | None,
        policy_version: str | None,
    ) -> bool:
        payload = self._safe_decode_payload(record)

        if record_type is not None:
            if payload.get("record_type") != record_type:
                return False

        if decision_id is not None:
            if payload.get("decision_id") != str(decision_id):
                return False

        if outcome_id is not None:
            if payload.get("outcome_id") != str(outcome_id):
                return False

        if request_id is not None:
            if payload.get("request_id") != str(request_id):
                return False

        if policy_version is not None:
            envelope_version = record.policy_version
            payload_version = payload.get("policy_version")

            if envelope_version != policy_version and payload_version != policy_version:
                return False

        return True

    def _safe_decode_payload(self, record: EvidenceRecord) -> dict[str, Any]:
        """
        Decodes payload_json into an object for filtering/export logic.

        Export should fail loudly on corrupted records rather than silently
        skipping them.
        """
        try:
            payload = self._recorder.decode_payload(record)
        except Exception as exc:
            raise ValueError(
                f"failed to decode evidence payload for record {record.evidence_id}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                f"decoded evidence payload for record {record.evidence_id} must be an object"
            )

        return payload