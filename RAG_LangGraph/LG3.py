from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str


def router(state: State):
    return state


def route_decider(state: State):
    question = state["question"]

    if "calculate" in question.lower():
        return "calculator"

    return "llm"


def calculator(state: State):
    print("Calculator Node")
    return state


def llm(state: State):
    print("LLM Node")
    return state


graph = StateGraph(State)

graph.add_node("router", router)
graph.add_node("calculator", calculator)
graph.add_node("llm", llm)

graph.add_edge(START, "router")

graph.add_conditional_edges(
    "router",
    route_decider,
    {
        "calculator": "calculator",
        "llm": "llm"
    }
)

graph.add_edge("calculator", END)
graph.add_edge("llm", END)

app = graph.compile()

#run the Graph
result = app.invoke({
    "question":"calculator"
})

print(result)