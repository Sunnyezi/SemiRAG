"""Adaptive RAG workflow with web-search fallback and answer grading."""

from pprint import pprint

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from semirag.utils.log_utils import log
from semirag.utils.text_encoding import normalize_terminal_text
from semirag.workflows.adaptive.generate_node2 import generate
from semirag.workflows.adaptive.grade_answer_chain import answer_grader_chain
from semirag.workflows.adaptive.grade_documents_node import grade_documents
from semirag.workflows.adaptive.grade_hallucinations_chain import hallucination_grader_chain
from semirag.workflows.adaptive.graph_state2 import GraphState
from semirag.workflows.adaptive.query_route_chain import question_router_chain
from semirag.workflows.adaptive.retriever_node import retrieve
from semirag.workflows.adaptive.transform_query_node import transform_query
from semirag.workflows.adaptive.web_search_node import web_search


def grade_generation_v_documents_and_question(state: GraphState) -> str:
    """Check grounding first, then check whether the answer addresses the question."""
    log.info("---检查生成内容是否存在幻觉---")
    score = hallucination_grader_chain.invoke(
        {"documents": state["documents"], "generation": state["generation"]}
    )
    if score.binary_score != "yes":
        log.info("---判定：生成内容未基于参考文档，将重新尝试---")
        return "not supported"

    log.info("---评估：生成回答与问题的匹配度---")
    score = answer_grader_chain.invoke(
        {"question": state["question"], "generation": state["generation"]}
    )
    if score.binary_score == "yes":
        log.info("---判定：生成内容准确回答问题---")
        return "useful"
    log.info("---判定：生成内容未能准确回答问题---")
    return "not useful"


def decide_to_generate(state: GraphState) -> str:
    """Generate from relevant documents or improve the query before retrying."""
    log.info("---ASSESS GRADED DOCUMENTS---")
    if state["documents"]:
        log.info("---决策：生成最终回答---")
        return "generate"
    if state.get("transform_count", 0) >= 2:
        log.info("---决策：两次改写后仍无相关文档，转为 web 查询---")
        return "web_search"
    log.info("---决策：改写查询后重试---")
    return "transform_query"


def route_question(state: GraphState) -> str:
    """Route semiconductor-domain questions to the vector store, others to web search."""
    log.info("---ROUTE QUESTION---")
    source = question_router_chain.invoke({"question": state["question"]})
    if source.datasource == "web_search":
        log.info("---路由到 web 搜索---")
        return "web_search"
    log.info("---路由到 RAG 系统---")
    return "vectorstore"


workflow = StateGraph(GraphState)
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_conditional_edges(
    START,
    route_question,
    {"web_search": "web_search", "vectorstore": "retrieve"},
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {"not supported": "generate", "useful": END, "not useful": "transform_query"},
)
workflow.add_edge("transform_query", "retrieve")

graph = workflow.compile()


def main() -> None:
    """Run the interactive adaptive RAG workflow."""
    while True:
        question = normalize_terminal_text(input("用户："))
        if question.lower() in ["q", "exit", "quit"]:
            print("对话结束，拜拜！")
            break

        for output in graph.stream({"question": question}):
            for key, value in output.items():
                pprint(f"Node '{key}':")
            pprint("\n---\n")

        pprint(value["generation"])


if __name__ == "__main__":
    main()
