from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer"""
    
    binary_score: bool = Field(
        description='Answer is grounded in the facts "yes", "no"'
    )

structued_llm_grader = llm.with_structured_output(GradeHallucinations)


system = """
    You are a grader assesing whether LLM generation is grounded in / supported by a set of documents.
    Give binary score "yes" or "no". "Yes" means that the answer it grounded in / supported by a set of documents.
"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system),
        ('human', 'Set of facts: \n\n {documents} \n\n LLM generations: {generation}'),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structued_llm_grader
