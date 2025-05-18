from datetime import datetime

from dotenv import load_dotenv

from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, RevisedAnswer

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
parser_json = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are expert researcher.
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improver your answer.
         """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.now().isoformat(),
)

first_responder = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer.",
) | llm.bind_tools(tools=[AnswerQuestion], tool_choice="auto")

revised_instruction = """
    Revise your previous answer using the inew information.
    You should use the previous critique to add important informaiton to your answer.
    You MUST include numerical citations in your revised answer to ensure it can be verified.
    Add a "References" setion to the bottom of your answer (which does not count towards the word limit). In form of:
        - [1] https://example.com
        - [2] https://example.com
    You should use the previous critique to remove superfluous information from your answer and make sure it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revised_instruction,
) | llm.bind_tools(tools=[RevisedAnswer], tool_choice="auto")
