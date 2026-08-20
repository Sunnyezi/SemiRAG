from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel

from semirag.models.all_llm import llm
from semirag.utils.choice_parser import parse_choice


# 数据模型 - 生成内容幻觉评分
class GradeHallucinations(BaseModel):
    """对生成回答中是否存在幻觉进行二元评分"""

    binary_score: str = Field(
        description="回答是否基于事实，取值为'yes'或'no'"
    )


# 提示词模板
system = """您是一个评估生成内容是否基于检索事实的评分器。\n
     'yes'表示回答是基于/支持于给定事实集的。
     只输出'yes'或'no'，不要输出其他内容。"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # 系统角色设定
        ("human", "事实集: \n\n {documents} \n\n 生成内容: {generation}"),  # 用户输入模板
    ]
)

def _parse_grade(response: object) -> GradeHallucinations:
    return GradeHallucinations(binary_score=parse_choice(response, ("yes", "no")))


# 构建幻觉检测工作流，避免依赖模型端 JSON Schema 输出能力。
hallucination_grader_chain = hallucination_prompt | llm | _parse_grade
