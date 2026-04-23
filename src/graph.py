from langchain_openai import ChatOpenAI
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from src.state import AgentState
from src.tools import mock_lead_capture
from src.rag import setup_retriever

# 1. Initialize LLM & RAG
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

retriever = setup_retriever()

class IntentClassification(BaseModel):
    intent: Literal["greeting", "inquiry", "high_intent"] = Field(
        description="Classify the user intent based on their latest message."
    )


class LeadExtraction(BaseModel):
    name: str | None = Field(default=None, description="The user's name if provided.")
    email: str | None = Field(default=None, description="The user's email if provided.")
    platform: str | None = Field(default=None, description="The user's content platform (e.g., YouTube, Instagram).")

#Graph Nodes
def detect_intent(state: AgentState):
    """Analyzes the latest message to determine the flow."""
    latest_message = state["messages"][-1].content
    structured_llm = llm.with_structured_output(IntentClassification)
    
    prompt = f"Analyze this user message: '{latest_message}'. Are they saying a casual greeting, asking about product/pricing (inquiry), or showing strong interest in buying/signing up (high_intent)?"
    result = structured_llm.invoke(prompt)
    
    return {"intent": result.intent}

def handle_greeting(state: AgentState):
    """Handles basic hellos."""
    response = llm.invoke([SystemMessage(content="You are a helpful assistant for AutoStream, a video editing SaaS. Give a short, friendly greeting and ask how you can help.")] + state["messages"])
    return {"messages": [response]}

def handle_inquiry(state: AgentState):
    """Uses RAG to answer questions."""
    latest_message = state["messages"][-1].content
    docs = retriever.invoke(latest_message)
    context = "\n".join([doc.page_content for doc in docs])
    
    sys_prompt = f"You are an assistant for AutoStream. Answer the user based ONLY on this context:\n{context}\nIf the answer is not in the context, say you don't know."
    response = llm.invoke([SystemMessage(content=sys_prompt)] + state["messages"])
    
    return {"messages": [response]}

def handle_lead(state: AgentState):
    """Extracts lead info, asks for missing info, or triggers the tool."""
    latest_message = state["messages"][-1].content
    
    # Extracting details mentioned in the new message
    extractor = llm.with_structured_output(LeadExtraction)
    extracted = extractor.invoke(f"Extract user details from this message if present: {latest_message}")
    
    current_name = extracted.name if extracted.name else state.get("lead_name")
    current_email = extracted.email if extracted.email else state.get("lead_email")
    current_platform = extracted.platform if extracted.platform else state.get("lead_platform")
    
    updates = {
        "lead_name": current_name,
        "lead_email": current_email,
        "lead_platform": current_platform
    }
    
    # Check have all 3
    if current_name and current_email and current_platform:
        # Trigger Tool
        success_msg = mock_lead_capture(current_name, current_email, current_platform)
        updates["messages"] = [AIMessage(content=success_msg)]
    else:
        
        missing = []
        if not current_name: missing.append("name")
        if not current_email: missing.append("email address")
        if not current_platform: missing.append("content platform (like YouTube or Instagram)")
        
        prompt = f"We'd love to get you set up! To proceed, I just need your {', '.join(missing)}."
        updates["messages"] = [AIMessage(content=prompt)]
        
    return updates


def route_intent(state: AgentState) -> str:
    return state["intent"]

workflow = StateGraph(AgentState)

workflow.add_node("detect_intent", detect_intent)
workflow.add_node("greeting", handle_greeting)
workflow.add_node("inquiry", handle_inquiry)
workflow.add_node("lead_capture", handle_lead)

workflow.add_edge(START, "detect_intent")
workflow.add_conditional_edges(
    "detect_intent",
    route_intent,
    {
        "greeting": "greeting",
        "inquiry": "inquiry",
        "high_intent": "lead_capture"
    }
)
workflow.add_edge("greeting", END)
workflow.add_edge("inquiry", END)
workflow.add_edge("lead_capture", END)

app = workflow.compile()