from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from tex.domain.decision import Decision
from tex.domain.evaluation import EvaluationRequest, EvaluationResponse
from tex.domain.evidence import EvidenceRecord
from tex.domain.policy import PolicySnapshot
from tex.engine.pdp import PDPResult, PolicyDecisionPoint
from tex.stores.decision_store import InMemoryDecisionStore
from tex.stores.policy_store import InMemoryPolicyStore
from tex.stores.precedent_store import InMemoryPrecedentStore


@runtime_checkable
class DecisionEvidenceRecorder(Protocol):
    """
    Narrow protocol for recorders capable of appending decision evidence.
    """

    def record_decision(
        self,
        decision: Decision,
        *,
        metadata: dict[str, object] | None = None,
    ) -> EvidenceRecord:
        """
        Persist an evidence envelope for a decision.
        """
        ...


@runtime_checkable
class DecisionPrecedentStore(Protocol):
    """
    Narrow protocol for stores that can retain decisions as precedents.
    """

    def save(self, decision: Decision) -> None:
        """
        Persist a decision so it can be retrieved later as precedent context.
        """
        ...


@dataclass(frozen=True, slots=True)
class EvaluateActionResult:
    """
    Application-layer result for a single Tex evaluation command.

    This wraps the public response with the internal decision, resolved policy,
    raw PDP result, and any evidence record created during execution.
    """

    response: EvaluationResponse
    decision: Decision
    policy: PolicySnapshot
    pdp_result: PDPResult
    evidence_record: EvidenceRecord | None = None


class EvaluateActionCommand:
    """
    Application service for evaluating one action through Tex.

    Responsibilities:
    - resolve the policy snapshot to use
    - run the PDP
    - validate request / policy / output alignment
    - persist the resulting decision
    - persist the decision as precedent context when configured
    - optionally append evidence

    Responsibilities intentionally excluded:
    - HTTP transport
    - policy activation workflows
    - outcome reporting
    - calibration
    """

    __slots__ = (
        "_pdp",
        "_policy_store",
        "_decision_store",
        "_precedent_store",
        "_evidence_recorder",
    )

    def __init__(
        self,
        *,
        pdp: PolicyDecisionPoint,
        policy_store: InMemoryPolicyStore,
        decision_store: InMemoryDecisionStore,
        precedent_store: DecisionPrecedentStore | None = None,
        evidence_recorder: DecisionEvidenceRecorder | None = None,
    ) -> None:
        self._pdp = pdp
        self._policy_store = policy_store
        self._decision_store = decision_store
        self._precedent_store = precedent_store
        self._evidence_recorder = evidence_recorder

    def execute(self, request: EvaluationRequest) -> EvaluateActionResult:
        """
        Evaluate a request, persist the decision, update precedent memory,
        and optionally record evidence.
        """
        policy = self._resolve_policy(request)
        pdp_result = self._pdp.evaluate(
            request=request,
            policy=policy,
        )

        self._validate_pdp_alignment(
            request=request,
            policy=policy,
            pdp_result=pdp_result,
        )

        decision = pdp_result.decision
        self._decision_store.save(decision)
        self._save_precedent(decision)

        evidence_record = None
        if self._evidence_recorder is not None:
            evidence_record = self._record_decision_evidence(
                decision=decision,
                request=request,
            )

        return EvaluateActionResult(
            response=pdp_result.response,
            decision=decision,
            policy=policy,
            pdp_result=pdp_result,
            evidence_record=evidence_record,
        )

    def _resolve_policy(self, request: EvaluationRequest) -> PolicySnapshot:
        """
        Resolve the policy snapshot for the request.

        Rules:
        - if request.policy_id is set, treat it as the requested policy version
        - otherwise, use the currently active policy
        """
        if request.policy_id is not None:
            requested_version = request.policy_id.strip()
            if not requested_version:
                raise ValueError("request.policy_id must not be blank when provided")

            try:
                return self._policy_store.require(requested_version)
            except KeyError as exc:
                raise LookupError(
                    f"requested policy version not found: {requested_version}"
                ) from exc

        try:
            return self._policy_store.require_active()
        except LookupError as exc:
            raise LookupError("no active policy is available for evaluation") from exc

    @staticmethod
    def _validate_pdp_alignment(
        *,
        request: EvaluationRequest,
        policy: PolicySnapshot,
        pdp_result: PDPResult,
    ) -> None:
        """
        Enforce basic integrity between the request, selected policy, and PDP output.
        """
        decision = pdp_result.decision
        response = pdp_result.response

        if decision.request_id != request.request_id:
            raise ValueError("pdp decision.request_id does not match evaluation request")

        if decision.policy_version != policy.version:
            raise ValueError("pdp decision.policy_version does not match selected policy")

        if response.decision_id != decision.decision_id:
            raise ValueError(
                "pdp response.decision_id does not match decision.decision_id"
            )

        if response.policy_version != decision.policy_version:
            raise ValueError(
                "pdp response.policy_version does not match decision.policy_version"
            )

        if response.verdict != decision.verdict:
            raise ValueError("pdp response.verdict does not match decision.verdict")

        if response.confidence != decision.confidence:
            raise ValueError("pdp response.confidence does not match decision.confidence")

        if response.final_score != decision.final_score:
            raise ValueError("pdp response.final_score does not match decision.final_score")

    def _save_precedent(self, decision: Decision) -> None:
        """
        Persist the evaluated decision into precedent memory when configured.

        This is the critical bridge that lets retrieval improve from live traffic
        instead of staying permanently empty.
        """
        store = self._precedent_store
        if store is None:
            return

        if not isinstance(store, DecisionPrecedentStore):
            raise TypeError("precedent_store must implement save(decision)")

        store.save(decision)

    def _record_decision_evidence(
        self,
        *,
        decision: Decision,
        request: EvaluationRequest,
    ) -> EvidenceRecord:
        """
        Record decision evidence using a narrow recorder protocol.
        """
        recorder = self._evidence_recorder
        if recorder is None:
            raise RuntimeError("evidence recorder is not configured")

        if not isinstance(recorder, DecisionEvidenceRecorder):
            raise TypeError(
                "evidence_recorder must implement record_decision("
                "decision, *, metadata=None)"
            )

        metadata: dict[str, object] = {
            "request_id": str(request.request_id),
            "request_channel": request.channel,
            "request_environment": request.environment,
            "request_action_type": request.action_type,
        }

        if request.recipient is not None:
            metadata["request_recipient"] = request.recipient
        if request.policy_id is not None:
            metadata["requested_policy_id"] = request.policy_id
        if request.metadata:
            metadata["request_metadata"] = dict(request.metadata)

        return recorder.record_decision(
            decision,
            metadata=metadata,
        )
