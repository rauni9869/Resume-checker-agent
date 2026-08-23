from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from resume_checker.graph.nodes import (
    critique_node,
    extract_node,
    report_node,
    score_node,
    validate_inputs_node,
)
from resume_checker.graph.state import GraphState
from resume_checker.schemas import AnalysisRequest, AnalysisResult

_GRAPH = None


def _route_after_validate(state: GraphState) -> str:
    return "report" if state.get("blocked") else "extract"


def _route_after_extract(state: GraphState) -> str:
    return "report" if state.get("blocked") else "score"


def build_graph():
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH
    builder = StateGraph(GraphState)
    builder.add_node("validate", validate_inputs_node)
    builder.add_node("extract", extract_node)
    builder.add_node("score", score_node)
    builder.add_node("critique", critique_node)
    builder.add_node("report", report_node)
    builder.add_edge(START, "validate")
    builder.add_conditional_edges(
        "validate", _route_after_validate, {"extract": "extract", "report": "report"}
    )
    builder.add_conditional_edges(
        "extract", _route_after_extract, {"score": "score", "report": "report"}
    )
    builder.add_edge("score", "critique")
    builder.add_edge("critique", "report")
    builder.add_edge("report", END)
    _GRAPH = builder.compile()
    return _GRAPH


def analyze_resume(
    request: AnalysisRequest,
    resume_path: str | None = None,
    include_specialists: bool = False,
) -> AnalysisResult:
    graph = build_graph()
    state = graph.invoke(
        {
            "candidate_id": request.candidate_id,
            "job_description": request.job_description,
            "resume_path": resume_path,
            "resume_text": request.resume_text,
            "recipient_email": str(request.recipient_email) if request.recipient_email else None,
            "send_email": request.send_email,
            "include_specialists": include_specialists,
            "guardrails": [],
            "blocked": False,
            "semantic_panel": [],
        }
    )
    if state.get("result"):
        return AnalysisResult.model_validate(state["result"])
    return AnalysisResult(
        ok=False,
        candidate_id=request.candidate_id,
        extraction=state.get("extraction"),
        ats=state.get("ats"),
        semantic_panel=list(state.get("semantic_panel") or []),
        critique=state.get("critique"),
        composite_score=state.get("composite_score"),
        guardrails=list(state.get("guardrails") or []),
        blocked=True,
        llm_backend=state.get("llm_backend") or "template",
        html_report=state.get("html_report"),
    )
