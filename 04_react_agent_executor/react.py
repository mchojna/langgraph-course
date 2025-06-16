from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num: float) -> float:
    """
    Returns a number times three
    
    Attributes:
        num (float): a number to triple
    
    Returns: the triple of the input number
    """
    
    return float(num) * 3

tools = [TavilySearch(max_results=1), triple]

llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0
).bind_tools(tools)