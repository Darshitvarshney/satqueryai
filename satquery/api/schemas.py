"""Request / response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    evidence_id: str | None = None
    agent: str | None = None
    task: str | None = None
    finding: Any = None
    verified_finding: str | None = None
    confidence: float = 0.0
    status: str = "ok"
    visual_evidence: list[str] = Field(default_factory=list)


class AnalyzeResult(BaseModel):
    query: str = ""
    final_answer: str = ""
    confidence: float = 0.0
    current_task: str | None = None
    temporal_mode: str | None = None
    modalities: list[str] = Field(default_factory=list)
    image_count: int = 0
    input_valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)
    retry_count: int = 0
    reflection: dict[str, Any] | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    execution_trace: list[Any] = Field(default_factory=list)
    duration_seconds: float | None = None

    @classmethod
    def from_state(cls, state: dict) -> "AnalyzeResult":
        evidence: list[EvidenceItem] = []
        for raw in state.get("evidence", []) or []:
            data = {k: v for k, v in raw.items() if k in EvidenceItem.model_fields}
            data["confidence"] = float(data.get("confidence") or 0.0)
            evidence.append(EvidenceItem(**data))

        try:
            confidence = float(state.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        return cls(
            query=state.get("query", ""),
            final_answer=state.get("final_answer", ""),
            confidence=confidence,
            current_task=state.get("current_task"),
            temporal_mode=state.get("temporal_mode"),
            modalities=state.get("modalities", []),
            image_count=state.get("image_count", 0),
            input_valid=state.get("input_valid", True),
            validation_errors=state.get("validation_errors", []),
            retry_count=state.get("retry_count", 0),
            reflection=state.get("reflection"),
            evidence=evidence,
            execution_trace=state.get("execution_trace", []),
            duration_seconds=state.get("duration_seconds"),
        )


class AnalyzeJsonRequest(BaseModel):
    query: str = Field(min_length=1)
    max_retries: int | None = Field(default=None, ge=0, le=5)

    optical_image_url: str | None = None
    sar_image_url: str | None = None
    image_t1_url: str | None = None
    image_t2_url: str | None = None

    optical_image_b64: str | None = None
    sar_image_b64: str | None = None
    image_t1_b64: str | None = None
    image_t2_b64: str | None = None


class JobSummary(BaseModel):
    job_id: str
    status: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None


class JobDetail(JobSummary):
    result: AnalyzeResult | None = None
