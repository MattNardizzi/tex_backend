from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from tex.domain.evidence import EvidenceRecord


@dataclass(frozen=True, slots=True)
class ChainVerificationIssue:
    """
    Describes a single integrity problem found during evidence-chain verification.
    """

    index: int
    record_hash: str | None
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ChainVerificationResult:
    """
    Result of verifying one or more evidence records.
    """

    is_valid: bool
    record_count: int
    issues: tuple[ChainVerificationIssue, ...]

    @property
    def issue_count(self) -> int:
        return len(self.issues)


def verify_evidence_chain(
    records: Iterable[EvidenceRecord],
) -> ChainVerificationResult:
    """
    Verifies a full append-only evidence chain.

    Checks:
    - each record's payload_sha256 matches payload_json
    - each record's record_hash matches payload_sha256 + previous_hash
    - each record links correctly to the prior record
    - the first record does not point backward to a nonexistent predecessor
    """
    normalized_records = tuple(records)
    issues: list[ChainVerificationIssue] = []

    if not normalized_records:
        return ChainVerificationResult(
            is_valid=True,
            record_count=0,
            issues=tuple(),
        )

    previous_record: EvidenceRecord | None = None

    for index, record in enumerate(normalized_records):
        issues.extend(_verify_record_integrity(record=record, index=index))
        issues.extend(
            _verify_chain_link(
                previous_record=previous_record,
                candidate_record=record,
                index=index,
            )
        )
        previous_record = record

    return ChainVerificationResult(
        is_valid=not issues,
        record_count=len(normalized_records),
        issues=tuple(issues),
    )


def verify_latest_link(
    previous_record: EvidenceRecord | None,
    candidate_record: EvidenceRecord,
) -> ChainVerificationResult:
    """
    Verifies only the newest record being appended to the chain.

    Checks:
    - candidate payload_sha256 integrity
    - candidate record_hash integrity
    - candidate previous_hash linkage against the prior record
    """
    issues: list[ChainVerificationIssue] = []

    issues.extend(_verify_record_integrity(record=candidate_record, index=0))
    issues.extend(
        _verify_chain_link(
            previous_record=previous_record,
            candidate_record=candidate_record,
            index=0,
        )
    )

    return ChainVerificationResult(
        is_valid=not issues,
        record_count=1,
        issues=tuple(issues),
    )


def _verify_record_integrity(
    *,
    record: EvidenceRecord,
    index: int,
) -> list[ChainVerificationIssue]:
    issues: list[ChainVerificationIssue] = []

    try:
        expected_payload_sha256 = _sha256_hex(record.payload_json)
        if record.payload_sha256 != expected_payload_sha256:
            issues.append(
                ChainVerificationIssue(
                    index=index,
                    record_hash=record.record_hash,
                    code="payload_sha256_mismatch",
                    message=(
                        "record payload_sha256 does not match the canonical hash "
                        "of payload_json"
                    ),
                )
            )
    except Exception as exc:
        issues.append(
            ChainVerificationIssue(
                index=index,
                record_hash=record.record_hash,
                code="payload_sha256_verification_error",
                message=(
                    "payload_sha256 verification raised "
                    f"{exc.__class__.__name__}"
                ),
            )
        )

    try:
        expected_record_hash = _build_record_hash(
            payload_sha256=record.payload_sha256,
            previous_hash=record.previous_hash,
        )
        if record.record_hash != expected_record_hash:
            issues.append(
                ChainVerificationIssue(
                    index=index,
                    record_hash=record.record_hash,
                    code="record_hash_mismatch",
                    message=(
                        "record record_hash does not match the canonical hash of "
                        "payload_sha256 + previous_hash"
                    ),
                )
            )
    except Exception as exc:
        issues.append(
            ChainVerificationIssue(
                index=index,
                record_hash=record.record_hash,
                code="record_hash_verification_error",
                message=f"record hash verification raised {exc.__class__.__name__}",
            )
        )

    try:
        decoded_payload = json.loads(record.payload_json)
        if not isinstance(decoded_payload, dict):
            issues.append(
                ChainVerificationIssue(
                    index=index,
                    record_hash=record.record_hash,
                    code="payload_json_not_object",
                    message="record payload_json must decode to a JSON object",
                )
            )
    except json.JSONDecodeError:
        issues.append(
            ChainVerificationIssue(
                index=index,
                record_hash=record.record_hash,
                code="payload_json_invalid",
                message="record payload_json is not valid JSON",
            )
        )
    except Exception as exc:
        issues.append(
            ChainVerificationIssue(
                index=index,
                record_hash=record.record_hash,
                code="payload_json_verification_error",
                message=f"payload_json verification raised {exc.__class__.__name__}",
            )
        )

    return issues


def _verify_chain_link(
    *,
    previous_record: EvidenceRecord | None,
    candidate_record: EvidenceRecord,
    index: int,
) -> list[ChainVerificationIssue]:
    issues: list[ChainVerificationIssue] = []

    if previous_record is None:
        if candidate_record.previous_hash is not None:
            issues.append(
                ChainVerificationIssue(
                    index=index,
                    record_hash=candidate_record.record_hash,
                    code="unexpected_previous_hash",
                    message="first record must not contain a previous_hash",
                )
            )
        return issues

    if candidate_record.previous_hash != previous_record.record_hash:
        issues.append(
            ChainVerificationIssue(
                index=index,
                record_hash=candidate_record.record_hash,
                code="chain_link_mismatch",
                message=(
                    "record previous_hash does not match the prior record's "
                    "record_hash"
                ),
            )
        )

    return issues


def _build_record_hash(
    *,
    payload_sha256: str,
    previous_hash: str | None,
) -> str:
    chain_input = _stable_json(
        {
            "payload_sha256": payload_sha256,
            "previous_hash": previous_hash,
        }
    )
    return _sha256_hex(chain_input)


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()