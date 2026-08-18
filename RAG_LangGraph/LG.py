from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class state(TypedDict):
    message:str

def node_a(state:state):
    print("A")

    return{
        "message":state["message"]+"->node a"
    }


def node_b(state:state):
    print("B")

    return{
        "message":state["message"]+"->node B"
    }

builder=StateGraph(state)

builder.add_node("node_a",node_a)
builder.add_node("node_b",node_b)

builder.add_edge(START,"node_a")
builder.add_edge("node_a","node_b")
builder.add_edge("node_b",END)


graph =builder.compile()


result = graph.invoke({
    "message":"START"
})

print(result)
