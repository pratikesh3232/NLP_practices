from datetime import datetime

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================
# 1. TOOLS
# ============================================================

@tool
def calculator(a: float, b: float, operation: str) -> float:
    """Perform basic arithmetic operations."""

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")

        return a / b

    else:
        raise ValueError(
            "Operation must be add, subtract, multiply, or divide"
        )


@tool
def get_weather(city: str) -> str:
    """Get mock weather information for a city."""

    weather_data = {
        "pune": "28°C, Sunny",
        "mumbai": "30°C, Humid",
        "delhi": "32°C, Clear",
        "bangalore": "24°C, Cloudy",
    }

    return weather_data.get(
        city.lower(),
        f"Weather data not available for {city}"
    )


@tool
def search_web(query: str) -> str:
    """Search for information. This is a mock search tool."""

    return (
        f"Mock search results for '{query}': "
        f"This is simulated search information."
    )


@tool
def get_datetime() -> str:
    """Get the current date and time."""

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# 2. PUT ALL TOOLS INTO A LIST
# ============================================================

tools = [
    calculator,
    get_weather,
    search_web,
    get_datetime,
]


# ============================================================
# 3. CREATE LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ============================================================
# 4. GIVE TOOLS TO LLM
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 5. CREATE TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 6. LLM NODE
# ============================================================

def call_llm(state: MessagesState):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# ============================================================
# 7. BUILD GRAPH
# ============================================================

builder = StateGraph(MessagesState)


# Add nodes
builder.add_node("llm", call_llm)
builder.add_node("tools", tool_node)


# ============================================================
# 8. EDGES
# ============================================================

# START → LLM
builder.add_edge(
    START,
    "llm"
)


# LLM → ToolNode OR END
builder.add_conditional_edges(
    "llm",
    tools_condition
)


# ToolNode → LLM
builder.add_edge(
    "tools",
    "llm"
)


# ============================================================
# 9. COMPILE GRAPH
# ============================================================

graph = builder.compile()


# ============================================================
# 10. RUN
# ============================================================

result = graph.invoke({
    "messages": [
        ("user", "What's the weather in Pune?")
    ]
})


# ============================================================
# 11. PRINT FINAL ANSWER
# ============================================================

print(
    result["messages"][-1].content
)