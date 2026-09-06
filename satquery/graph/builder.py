"""Assemble and compile the SatQuery LangGraph.

Flow::

    START -> context_manager -> input_validation
        input_validation --(invalid)--> answer_synthesis -> END
        input_validation --(ok)------> supervisor
        supervisor --(specialist_router)--> {image_analysis | change_detection
                                             | cross_modal | geo_spatial | retrieval}
        <specialist> -> evidence_pool -> verification -> reflection
        reflection --(retry)------> retry -> supervisor
        reflection --(validated)--> answer_synthesis -> END
"""

from __future__ import annotations

from functools import lru_cache

from satquery.graph.nodes.evidence import evidence_pool_node, verification_node
from satquery.graph.nodes.preprocessing import context_manager, input_validation
from satquery.graph.nodes.reflection import reflection_node, retry_node
from satquery.graph.nodes.routers import (
    reflection_router,
    specialist_router,
    validation_router,
)
from satquery.graph.nodes.specialists import (
    change_detection_node,
    cross_modal_agent_node,
    geo_spatial_agent_node,
    image_analysis_node,
    retrieval_node,
)
from satquery.graph.nodes.supervisor import supervisor_node
from satquery.graph.nodes.synthesis import answer_synthesis_node
from satquery.graph.state import SatQueryState

_SPECIALIST_NODES = {
    "image_analysis": image_analysis_node,
    "change_detection": change_detection_node,
    "cross_modal": cross_modal_agent_node,
    "geo_spatial": geo_spatial_agent_node,
    "retrieval": retrieval_node,
}


def build_graph():
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(SatQueryState)

    builder.add_node("context_manager", context_manager)
    builder.add_node("input_validation", input_validation)
    builder.add_node("supervisor", supervisor_node)
    for name, fn in _SPECIALIST_NODES.items():
        builder.add_node(name, fn)
    builder.add_node("evidence_pool", evidence_pool_node)
    builder.add_node("verification", verification_node)
    builder.add_node("reflection", reflection_node)
    builder.add_node("retry", retry_node)
    builder.add_node("answer_synthesis", answer_synthesis_node)

    builder.add_edge(START, "context_manager")
    builder.add_edge("context_manager", "input_validation")
    builder.add_conditional_edges(
        "input_validation",
        validation_router,
        {"ok": "supervisor", "invalid": "answer_synthesis"},
    )
    builder.add_conditional_edges(
        "supervisor",
        specialist_router,
        {name: name for name in _SPECIALIST_NODES},
    )
    for name in _SPECIALIST_NODES:
        builder.add_edge(name, "evidence_pool")
    builder.add_edge("evidence_pool", "verification")
    builder.add_edge("verification", "reflection")
    builder.add_conditional_edges(
        "reflection",
        reflection_router,
        {"retry": "retry", "validated": "answer_synthesis"},
    )
    builder.add_edge("retry", "supervisor")
    builder.add_edge("answer_synthesis", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


def reset_graph() -> None:
    get_graph.cache_clear()
