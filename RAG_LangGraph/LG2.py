from typing import TypedDict
from langgraph.graph import StateGraph,START,END

class State(TypedDict):
    q:str
    a:str


#Node

def Store_que(state:State):
    q = state["q"]

    print("Question : ",q)

    return{
        "q" : q
    }

#Node

def genrate_ans(state:State):
    q =state["q"]
    a = f"you asked:{q}"

    return{
        "a":a
    }

#builder

builder = StateGraph(State)

builder.add_node("store q",Store_que)
builder.add_node("genrate a",genrate_ans)

builder.add_edge(START,"store q")
builder.add_edge("store q","genrate a")
builder.add_edge("genrate a",END)

graph = builder.compile()


#run the Graph
result = graph.invoke({
    "q":"what is langGraph"
})

print(result)


