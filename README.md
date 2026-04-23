# AutoStream - Social-to-Lead Agentic Workflow

An intelligent conversational agent built for AutoStream that handles customer inquiries via RAG and converts high-intent users into captured leads using a stateful LangGraph architecture.

##  How to Run Locally

1. **Clone the repository.**
2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

## Install dependencies:

pip install -r requirements.txt



## ⚠️ Note on API Rate Limits (429 Errors)
This agent is built using the free tier of Google AI Studio (Gemini 2.0 Flash). Because this is a stateful LangGraph agent, a single user input triggers multiple rapid LLM calls under the hood (Intent Classification -> Entity Extraction -> RAG/Response Generation). 

As a result, the agent operates very close to Google's free-tier rate limit of **15 Requests Per Minute**. 
* To prevent `429 RESOURCE_EXHAUSTED` crashes during testing, I have implemented a `time.sleep(2)` artificial delay in the execution loop. 
* If testing manually, please allow a few seconds between messages. In a production environment, this architecture would be backed by a paid, high-throughput API tier (like GPT-4o or Gemini Pro), eliminating these delays entirely.