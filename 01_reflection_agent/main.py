from typing import List, Sequence
from IPython.display import Image, display

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import generation_chain, reflection_chain

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    return generation_chain.invoke({"messages": state})


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflection_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return "__end__"
    else:
        return "__reflect__"


builder.add_conditional_edges(
    GENERATE,
    should_continue,
    {
        "__end__": END,
        "__reflect__": REFLECT,
    },
)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))


if __name__ == "__main__":
    inputs = HumanMessage(
        content="""
            Make this tweet better:
             literally cannot wait til I hit 30, get a lil house, adopt 20 cats, & finally fulfill my lifelong dream of becoming a crazy cat lady.
        """
    )
    response = graph.invoke(inputs)
    print(response)
