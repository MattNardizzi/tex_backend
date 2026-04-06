from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import UUID

from tex.evidence.exporter import EvidenceExportBundle, EvidenceExporter


@runtime_checkable
class BundleCapableExporter(Protocol):
    """
    Narrow exporter protocol for command-layer bundle exports.
    """

    def build_bundle(
        self,
        *,
        export_name: str = "tex-evidence-bundle",
        verify_chain: bool = True,
    ) -> EvidenceExportBundle:
        ...

    def export_json(
        self,
        path: str | Path,
        *,
        export_name: str = "tex-evidence-bundle",
        verify_chain: bool = True,
        indent: int = 2,
    ) -> Path:
        ...

    def export_jsonl(
        self,
        path: str | Path,
    ) -> Path:
        ...

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
        ...

    def filter_records(
        self,
        *,
        record_type: str | None = None,
        decision_id: str | UUID | None = None,
        outcome_id: str | UUID | None = None,
        request_id: str | UUID | None = None,
        policy_version: str | None = None,
    ) -> tuple:
        ...


@dataclass(frozen=True, slots=True)
class ExportBundleResult:
    """
    Application-layer result for exporting evidence bundles.
    """

    output_path: Path
    bundle: EvidenceExportBundle | None
    export_format: str


class ExportBundleCommand:
    """
    Application service for exporting Tex evidence artifacts.

    Responsibilities:
    - build full evidence bundles
    - export wrapped JSON bundles
    - export raw JSONL records
    - export filtered JSON bundles

    This command stays thin and delegates packaging mechanics to the exporter.
    """

    __slots__ = ("_exporter",)

    def __init__(self, *, exporter: EvidenceExporter) -> None:
        if not isinstance(exporter, BundleCapableExporter):
            raise TypeError(
                "exporter must implement bundle/json/jsonl/filtered export methods"
            )
        self._exporter = exporter

    def export_json(
        self,
        *,
        path: str | Path,
        export_name: str = "tex-evidence-bundle",
        verify_chain: bool = True,
        indent: int = 2,
    ) -> ExportBundleResult:
        """
        Exports a full JSON evidence bundle and returns the same in-memory bundle
        metadata that corresponds to the exported file.
        """
        bundle = self._exporter.build_bundle(
            export_name=export_name,
            verify_chain=verify_chain,
        )
        output_path = self._exporter.export_json(
            path,
            export_name=export_name,
            verify_chain=verify_chain,
            indent=indent,
        )

        return ExportBundleResult(
            output_path=output_path,
            bundle=bundle,
            export_format="json",
        )

    def export_jsonl(
        self,
        *,
        path: str | Path,
    ) -> ExportBundleResult:
        """
        Exports raw evidence records as JSONL.

        JSONL does not wrap records in an in-memory EvidenceExportBundle, so the
        returned bundle field is None.
        """
        output_path = self._exporter.export_jsonl(path)

        return ExportBundleResult(
            output_path=output_path,
            bundle=None,
            export_format="jsonl",
        )

    def export_filtered_json(
        self,
        *,
        path: str | Path,
        record_type: str | None = None,
        decision_id: str | UUID | None = None,
        outcome_id: str | UUID | None = None,
        request_id: str | UUID | None = None,
        policy_version: str | None = None,
        export_name: str = "tex-evidence-filtered-bundle",
        verify_chain: bool = False,
        indent: int = 2,
    ) -> ExportBundleResult:
        """
        Exports a filtered JSON bundle.

        Filtered subsets are not guaranteed to remain a contiguous valid chain,
        so verify_chain defaults to False.
        """
        output_path = self._exporter.export_filtered_json(
            path,
            record_type=record_type,
            decision_id=decision_id,
            outcome_id=outcome_id,
            request_id=request_id,
            policy_version=policy_version,
            export_name=export_name,
            verify_chain=verify_chain,
            indent=indent,
        )

        filtered_records = self._exporter.filter_records(
            record_type=record_type,
            decision_id=decision_id,
            outcome_id=outcome_id,
            request_id=request_id,
            policy_version=policy_version,
        )

        if verify_chain:
            from tex.evidence.chain import verify_evidence_chain

            verification = verify_evidence_chain(filtered_records)
        else:
            from tex.evidence.chain import ChainVerificationResult

            verification = ChainVerificationResult(
                is_valid=True,
                record_count=len(filtered_records),
                issues=tuple(),
            )

        bundle = EvidenceExportBundle(
            export_name=export_name,
            record_count=len(filtered_records),
            is_chain_valid=verification.is_valid,
            verification=verification,
            records=tuple(filtered_records),
        )

        return ExportBundleResult(
            output_path=output_path,
            bundle=bundle,
            export_format="json",
        )