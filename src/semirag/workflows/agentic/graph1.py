"""Tool-calling Agentic RAG workflow."""

import uuid
from typing import Literal

from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from semirag.models.all_llm import llm
from semirag.retrieval.retriever_tools import retriever_tool
from semirag.utils.log_utils import log
from semirag.utils.print_utils import _print_event
from semirag.workflows.agentic.agent_node import agent_node
from semirag.workflows.agentic.generate_node import generate
from semirag.workflows.agentic.get_human_message import get_last_human_message
from semirag.workflows.agentic.graph_state1 import AgentState, Grade
from semirag.workflows.agentic.rewrite_node import rewrite


def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    """Judge whether the retrieved documents are relevant to the user question."""
    log.info("---检查 document 的相关性---")
    llm_with_structured = llm.with_structured_output(Grade)
    prompt = PromptTemplate(
        template="""你是一个评估检索文档与用户问题相关性的评分器。\n
这是检索到的文档：\n\n {context} \n\n
这是用户的问题：{question} \n
如果文档包含与用户问题相关的关键词或语义含义，则评为相关。
给出二元评分 'yes' 或 'no' 来表示文档是否与问题相关。""",
        input_variables=["context", "question"],
    )
    chain = prompt | llm_with_structured

    messages = state["messages"]
    question = get_last_human_message(messages).content
    documents = messages[-1].content
    score = chain.invoke({"question": question, "context": documents}).binary_score

    if score == "yes":
        log.info("---输出：文档相关---")
        return "generate"
    log.info("---输出：文档不相关，改写问题后重试---")
    return "rewrite"


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("rewrite", "agent")
workflow.add_edge("generate", END)

graph = workflow.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": str(uuid.uuid4())}}


def main() -> None:
    """Run the interactive tool-calling RAG workflow."""
    printed = set()
    while True:
        question = input("用户：")
        if question.lower() in ["q", "exit", "quit"]:
            log.info("对话结束，拜拜！")
            break

        events = graph.stream(
            {"messages": [("user", question)]},
            config=config,
            stream_mode="values",
        )
        for event in events:
            _print_event(event, printed)


if __name__ == "__main__":
    main()
