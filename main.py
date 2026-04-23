
import os
import time
from dotenv import load_dotenv

# os.environ["OPENAI_API_KEY"] = ""
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.graph import app



def print_separator():
    print("-" * 50)

def main():
    print("AutoStream Agent is running! (Type 'quit' to exit)")
    print_separator()
    
    config = {"configurable": {"thread_id": "1"}}
    state = {
        "messages": [],
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None
    }
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        state["messages"].append(HumanMessage(content=user_input))
        
        # Stream graph updates
        for output in app.stream(state, config):
            for node_name, node_state in output.items():
                if "messages" in node_state:
                    latest_msg = node_state["messages"][-1].content
                    print(f"Agent: {latest_msg}")
                    
                # Update local state loop
                state.update(node_state)
        time.sleep(3)
if __name__ == "__main__":
    main()