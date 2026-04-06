from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from tex.domain.evaluation import EvaluationRequest
from tex.domain.outcome import OutcomeKind, OutcomeRecord
from tex.main import build_runtime


def determine_outcome_for_decision(decision_verdict: str) -> tuple[OutcomeKind, bool]:
    if decision_verdict == "PERMIT":
        return OutcomeKind.RELEASED, True

    if decision_verdict == "FORBID":
        return OutcomeKind.BLOCKED, False

    if decision_verdict == "ABSTAIN":
        return OutcomeKind.ESCALATED, True

    raise ValueError(f"unsupported decision verdict: {decision_verdict}")


def main() -> None:
    evidence_path = Path("var/tex/evidence/smoke_test.jsonl")
    full_bundle_path = Path("var/tex/exports/smoke_bundle.json")
    filtered_bundle_path = Path("var/tex/exports/smoke_bundle_filtered.json")

    runtime = build_runtime(evidence_path=evidence_path)

    request = EvaluationRequest(
        request_id=uuid4(),
        action_type="outbound_email",
        content=(
            "Please send the customer our pricing sheet and confirm "
            "we can guarantee 99.99% uptime."
        ),
        recipient="customer@example.com",
        channel="email",
        environment="production",
        metadata={"source": "smoke-test"},
        policy_id=None,
    )

    eval_result = runtime.evaluate_action_command.execute(request)
    decision = eval_result.decision

    print("\n=== EVALUATION RESULT ===")
    print("decision_id:", decision.decision_id)
    print("request_id:", decision.request_id)
    print("verdict:", decision.verdict.value)
    print("confidence:", decision.confidence)
    print("final_score:", decision.final_score)
    print("policy_version:", decision.policy_version)
    print("decision_evidence_recorded:", eval_result.evidence_record is not None)

    outcome_kind, was_safe = determine_outcome_for_decision(decision.verdict.value)

    outcome = OutcomeRecord(
        decision_id=decision.decision_id,
        request_id=decision.request_id,
        verdict=decision.verdict,
        outcome_kind=outcome_kind,
        was_safe=was_safe,
        human_override=False,
        summary="Smoke test outcome.",
        reporter="smoke-test",
    )

    outcome_result = runtime.report_outcome_command.execute(outcome)

    print("\n=== OUTCOME RESULT ===")
    print("outcome_id:", outcome_result.outcome.outcome_id)
    print("decision_id:", outcome_result.outcome.decision_id)
    print("request_id:", outcome_result.outcome.request_id)
    print("outcome_kind:", outcome_result.outcome.outcome_kind.value)
    print("label:", outcome_result.outcome.label.value)
    print("classification:", outcome_result.classification.classification)
    print("outcome_evidence_recorded:", outcome_result.evidence_record is not None)

    full_bundle_result = runtime.export_bundle_command.export_json(
        path=full_bundle_path,
        export_name="smoke-test-bundle",
        verify_chain=True,
    )

    print("\n=== FULL BUNDLE EXPORT ===")
    print("output_path:", full_bundle_result.output_path)
    print(
        "bundle_record_count:",
        full_bundle_result.bundle.record_count if full_bundle_result.bundle else None,
    )
    print(
        "chain_valid:",
        full_bundle_result.bundle.is_chain_valid if full_bundle_result.bundle else None,
    )

    filtered_bundle_result = runtime.export_bundle_command.export_filtered_json(
        path=filtered_bundle_path,
        decision_id=str(decision.decision_id),
        export_name="smoke-test-filtered",
        verify_chain=False,
    )

    print("\n=== FILTERED BUNDLE EXPORT ===")
    print("output_path:", filtered_bundle_result.output_path)
    print(
        "bundle_record_count:",
        filtered_bundle_result.bundle.record_count
        if filtered_bundle_result.bundle
        else None,
    )

    print("\n=== STORE COUNTS ===")
    print("decisions:", len(runtime.decision_store))
    print("outcomes:", len(runtime.outcome_store))

    print("\n=== FILES ===")
    print("evidence_file:", evidence_path.resolve())
    print("full_bundle:", full_bundle_path.resolve())
    print("filtered_bundle:", filtered_bundle_path.resolve())

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()