from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel

from semirag.models.all_llm import llm
from semirag.utils.choice_parser import parse_choice


# 数据模型 - 回答质量评分
class GradeAnswer(BaseModel):
    """评估回答是否解决用户问题的二元评分模型"""

    binary_score: str = Field(
        description="回答是否解决了问题，取值为'yes'或'no'"
    )


# 提示词模板
system = """您是一个评估回答是否解决用户问题的评分器。\n
     'yes'表示回答确实解决了该问题。
     只输出'yes'或'no'，不要输出其他内容。"""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # 系统角色设定
        ("human", "用户问题: \n\n {question} \n\n 生成回答: {generation}"),  # 用户输入模板
    ]
)

def _parse_grade(response: object) -> GradeAnswer:
    return GradeAnswer(binary_score=parse_choice(response, ("yes", "no")))


# 构建回答质量评估工作流，避免依赖模型端 JSON Schema 输出能力。
answer_grader_chain = answer_prompt | llm | _parse_grade
