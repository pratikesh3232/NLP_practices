from typing import TypedDict
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode


#State

class State(TypedDict):
    count:int
    message:str


# think node

def think(state:State):
    count = state['count']+1
    print(f"Thinking... {count}")

    return{
        "count":count,
        "message":f"Completed iteration{count}"
    }

# Check Node

def check(state:State):
    print(f"Checking iteration{state['count']}")

    return state


# Decide loop
def should_continue(state:State):
    if state["count"] < 3:
        print("ned more? YES-> loop")
        return "continue"

    print("need more? no -> Finish")
    return "finish"

#Graph
graph = StateGraph(State)

#add node
graph.add_node("think",think)
graph.add_node("check",check)

# 6. Connect the Nodes
# --------------------------------------------------

graph.add_edge(START, "think")

graph.add_edge("think", "check")


# 7. Conditional Routing
# --------------------------------------------------

graph.add_conditional_edges(
    "check",
    should_continue,
    {
        "continue": "think",
        "finish": END
    }
)



# 8. Compile the Graph
# --------------------------------------------------

app = graph.compile()


# 9. Run the Graph
# --------------------------------------------------

result = app.invoke({
    "count": 0,
    "message": ""
})


print("\nFinal State:")
print(result)