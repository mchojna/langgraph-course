from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class GradeAnsewr(BaseModel):
    binary_score: bool = Field(
        description='Answer addresses the question: "yes" or "no"'
    )

llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0
)
structured_llm_grader = llm.with_structured_output(GradeAnsewr)

system = """
    You are a grader assessing whether an answer addresses / resolves a question.
    Give a binary "yes" or "no". Yes means that the answer resolves the question.
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generated: {generation}")
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
