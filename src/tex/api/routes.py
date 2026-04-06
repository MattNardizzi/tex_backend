from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

from fastapi import APIRouter, HTTPException, Request, status

from tex.api.schemas import (
    ActivatePolicyRequestDTO,
    ActivatePolicyResponseDTO,
    CalibratePolicyRequestDTO,
    CalibratePolicyResponseDTO,
    EvaluateRequestDTO,
    EvaluateResponseDTO,
    ExportBundleRequestDTO,
    ExportBundleResponseDTO,
    ReportOutcomeRequestDTO,
    ReportOutcomeResponseDTO,
)
from tex.commands.activate_policy import ActivatePolicyCommand
from tex.commands.calibrate_policy import CalibratePolicyCommand
from tex.commands.evaluate_action import EvaluateActionCommand
from tex.commands.export_bundle import ExportBundleCommand
from tex.commands.report_outcome import ReportOutcomeCommand


@runtime_checkable
class SupportsExecuteEvaluate(Protocol):
    def execute(self, request: Any) -> Any:
        """Executes one Tex evaluation request."""


@runtime_checkable
class SupportsExecuteOutcome(Protocol):
    def execute(self, outcome: Any) -> Any:
        """Executes one Tex outcome-reporting request."""


@runtime_checkable
class SupportsExecuteActivatePolicy(Protocol):
    def execute(self, version: str) -> Any:
        """Activates one policy version."""


@runtime_checkable
class SupportsExecuteCalibratePolicy(Protocol):
    def execute(
        self,
        *,
        source_policy_version: str | None = None,
        classifications: tuple[Any, ...] | list[Any],
        new_version: str | None = None,
        save: bool = False,
        activate: bool = False,
        metadata_updates: dict[str, object] | None = None,
    ) -> Any:
        """Runs one calibration pass."""


@runtime_checkable
class SupportsExportBundle(Protocol):
    def export_json(
        self,
        *,
        path: str,
        export_name: str = "tex-evidence-bundle",
        verify_chain: bool = True,
        indent: int = 2,
    ) -> Any:
        """Exports a full JSON evidence bundle."""

    def export_jsonl(
        self,
        *,
        path: str,
    ) -> Any:
        """Exports raw JSONL evidence records."""

    def export_filtered_json(
        self,
        *,
        path: str,
        record_type: str | None = None,
        decision_id: str | None = None,
        outcome_id: str | None = None,
        export_name: str = "tex-evidence-filtered-bundle",
        verify_chain: bool = False,
        indent: int = 2,
    ) -> Any:
        """Exports a filtered JSON evidence bundle."""


router = APIRouter(tags=["tex"])


@router.get(
    "/health",
    summary="Health check",
)
def health_check() -> dict[str, str]:
    """Simple liveness endpoint for local development and smoke testing."""
    return {"status": "ok"}


@router.post(
    "/evaluate",
    response_model=EvaluateResponseDTO,
    status_code=status.HTTP_200_OK,
    summary="Evaluate one action through Tex",
)
def evaluate_action(
    payload: EvaluateRequestDTO,
    request: Request,
) -> EvaluateResponseDTO:
    """
    Evaluates one action request through Tex and returns the public response.
    """
    command = _get_evaluate_action_command(request)
    domain_request = payload.to_domain()

    try:
        result = command.execute(domain_request)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except TypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return EvaluateResponseDTO.from_command_result(result)


@router.post(
    "/outcomes",
    response_model=ReportOutcomeResponseDTO,
    status_code=status.HTTP_200_OK,
    summary="Report an observed outcome for a prior Tex decision",
)
def report_outcome(
    payload: ReportOutcomeRequestDTO,
    request: Request,
) -> ReportOutcomeResponseDTO:
    """
    Records what happened after a prior Tex decision and returns the resulting
    outcome classification.
    """
    command = _get_report_outcome_command(request)
    outcome = payload.to_domain()

    try:
        result = command.execute(outcome)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except TypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return ReportOutcomeResponseDTO.from_command_result(result)


@router.post(
    "/policies/activate",
    response_model=ActivatePolicyResponseDTO,
    status_code=status.HTTP_200_OK,
    summary="Activate a stored policy version",
)
def activate_policy(
    payload: ActivatePolicyRequestDTO,
    request: Request,
) -> ActivatePolicyResponseDTO:
    """
    Activates a stored policy version and returns the previous/next active state.
    """
    command = _get_activate_policy_command(request)

    try:
        result = command.execute(payload.version)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except TypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return ActivatePolicyResponseDTO.from_command_result(result)


@router.post(
    "/policies/calibrate",
    response_model=CalibratePolicyResponseDTO,
    status_code=status.HTTP_200_OK,
    summary="Run a calibration pass against classified outcomes",
)
def calibrate_policy(
    payload: CalibratePolicyRequestDTO,
    request: Request,
) -> CalibratePolicyResponseDTO:
    """
    Runs one calibration pass from already-classified outcomes.

    This route keeps calibration explicit. It does not search the full system
    for candidate outcomes on the caller's behalf.
    """
    command = _get_calibrate_policy_command(request)

    try:
        result = command.execute(
            source_policy_version=payload.source_policy_version,
            classifications=payload.to_domain_classifications(),
            new_version=payload.new_version,
            save=payload.save,
            activate=payload.activate,
            metadata_updates=payload.metadata_updates,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except TypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return CalibratePolicyResponseDTO.from_command_result(result)


@router.post(
    "/evidence/export",
    response_model=ExportBundleResponseDTO,
    status_code=status.HTTP_200_OK,
    summary="Export Tex evidence artifacts",
)
def export_bundle(
    payload: ExportBundleRequestDTO,
    request: Request,
) -> ExportBundleResponseDTO:
    """
    Exports Tex evidence artifacts in either wrapped JSON or raw JSONL form.
    """
    command = _get_export_bundle_command(request)

    try:
        if payload.export_format == "jsonl":
            result = command.export_jsonl(path=payload.path)
        elif (
            payload.record_type is not None
            or payload.decision_id is not None
            or payload.outcome_id is not None
        ):
            result = command.export_filtered_json(
                path=payload.path,
                record_type=payload.record_type,
                decision_id=payload.decision_id,
                outcome_id=payload.outcome_id,
                export_name=payload.export_name,
                verify_chain=payload.verify_chain,
                indent=payload.indent,
            )
        else:
            result = command.export_json(
                path=payload.path,
                export_name=payload.export_name,
                verify_chain=payload.verify_chain,
                indent=payload.indent,
            )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except TypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"failed to export evidence bundle: {exc}",
        ) from exc

    return ExportBundleResponseDTO.from_command_result(result)


def build_api_router() -> APIRouter:
    """Convenience constructor for Tex's FastAPI router."""
    return router


def _get_evaluate_action_command(request: Request) -> SupportsExecuteEvaluate:
    command = _require_app_state_attr(request, "evaluate_action_command")
    _assert_protocol(
        value=command,
        protocol=SupportsExecuteEvaluate,
        attribute_name="evaluate_action_command",
    )
    return cast(SupportsExecuteEvaluate, command)


def _get_report_outcome_command(request: Request) -> SupportsExecuteOutcome:
    command = _require_app_state_attr(request, "report_outcome_command")
    _assert_protocol(
        value=command,
        protocol=SupportsExecuteOutcome,
        attribute_name="report_outcome_command",
    )
    return cast(SupportsExecuteOutcome, command)


def _get_activate_policy_command(
    request: Request,
) -> SupportsExecuteActivatePolicy:
    command = _require_app_state_attr(request, "activate_policy_command")
    _assert_protocol(
        value=command,
        protocol=SupportsExecuteActivatePolicy,
        attribute_name="activate_policy_command",
    )
    return cast(SupportsExecuteActivatePolicy, command)


def _get_calibrate_policy_command(
    request: Request,
) -> SupportsExecuteCalibratePolicy:
    command = _require_app_state_attr(request, "calibrate_policy_command")
    _assert_protocol(
        value=command,
        protocol=SupportsExecuteCalibratePolicy,
        attribute_name="calibrate_policy_command",
    )
    return cast(SupportsExecuteCalibratePolicy, command)


def _get_export_bundle_command(request: Request) -> SupportsExportBundle:
    command = _require_app_state_attr(request, "export_bundle_command")
    _assert_protocol(
        value=command,
        protocol=SupportsExportBundle,
        attribute_name="export_bundle_command",
    )
    return cast(SupportsExportBundle, command)


def _require_app_state_attr(request: Request, name: str) -> object:
    if not hasattr(request.app.state, name):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Tex app is missing required state dependency: {name}. "
                "Initialize the command stack before serving requests."
            ),
        )
    return getattr(request.app.state, name)


def _assert_protocol(
    *,
    value: object,
    protocol: type[Protocol],
    attribute_name: str,
) -> None:
    if not isinstance(value, protocol):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Tex app state dependency {attribute_name!r} does not satisfy "
                f"the required interface: {protocol.__name__}"
            ),
        )