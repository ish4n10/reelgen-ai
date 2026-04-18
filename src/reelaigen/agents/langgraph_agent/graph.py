from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import GraphNodes
from .state import PDFContentAgentState


def build_graph(nodes: GraphNodes):
    graph = StateGraph(PDFContentAgentState)
    graph.add_node("initialize", nodes.initialize)
    graph.add_node("parse_pdf", nodes.parse_pdf)
    # graph.add_node("algorithm_parser", nodes.algorithm_parser)
    graph.add_node("content_parser", nodes.content_parser_node)
    graph.add_node("script_writer", nodes.script_writer_node)
    graph.add_node("visual_planner", nodes.visual_planner_node)
    graph.add_node("summary", nodes.summary)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "parse_pdf")
    # graph.add_edge("parse_pdf", "algorithm_parser")
    graph.add_edge("parse_pdf", "content_parser")
    # graph.add_edge("algorithm_parser", "content_parser")
    graph.add_edge("content_parser", "script_writer")
    graph.add_edge("script_writer", "visual_planner")
    graph.add_edge("visual_planner", "summary")
    graph.add_edge("summary", END)
    return graph.compile(checkpointer=MemorySaver())
