import os
import asyncio
from typing import TypedDict, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient

# 1. State definition
class AgentState(TypedDict):
    recipient_email: str
    recipient_name: str
    context: str
    subject: str
    body: str
    feedback: str
    passed_review: bool
    attempts: int

# 2. Nodes implementation
def draft_node(state: AgentState):
    print("\n--- [NODE: DRAFT] Drafting cold email ---")
    attempts = state.get("attempts", 0)
    
    # Initialize Ollama LLM
    llm = Ollama(model="qwen2.5:1.5b")
    
    prompt = f"""You are an expert cold email writer. Write a highly personalized, compelling cold email based on the recipient's name, context, and any previous feedback. Output your response exactly as:
Subject: <subject line>

Body:
<body content>

Recipient Name: {state["recipient_name"]}
Context: {state["context"]}
Previous Feedback: {state.get("feedback", "None")}
Attempts: {attempts}
"""
    
    content = llm.invoke(prompt).strip()
    
    # Parse subject and body
    subject = "Cold Email"
    body = content
    if "Subject:" in content and "Body:" in content:
        parts = content.split("Body:", 1)
        subject_part = parts[0].replace("Subject:", "").strip()
        body_part = parts[1].strip()
        subject = subject_part
        body = body_part
        
    print(f"Draft Subject: {subject}")
    print(f"Draft Body Preview: {body[:100]}...")
    
    return {
        "subject": subject,
        "body": body,
        "attempts": attempts + 1
    }

def review_node(state: AgentState):
    print("\n--- [NODE: REVIEW] Reviewing draft email ---")
    body = state.get("body", "")
    
    # Check length
    word_count = len(body.split())
    print(f"Word count: {word_count}")
    
    # We require at least 50 words and no placeholders
    too_short = word_count < 50
    has_placeholders = "[insert" in body.lower() or "bracket" in body.lower() or "<" in body or "{" in body
    
    # Also evaluate quality via LLM
    llm = Ollama(model="qwen2.5:1.5b")
    review_prompt = f"""You are a critical email reviewer. Assess if this cold email draft is too generic, unprofessional, or contains template variables. Reply with 'PASSED' or a clear reason for revision.

Subject: {state["subject"]}
Body: {body}
"""
    
    review_response = llm.invoke(review_prompt).strip()
    print(f"Review response: {review_response}")
    
    if too_short:
        feedback = "The email draft is too short. Please write a more detailed, substantial email."
        passed = False
    elif has_placeholders:
        feedback = "The email contains template placeholders or bracketed variables. Make sure it is fully filled in."
        passed = False
    elif "PASSED" not in review_response.upper():
        feedback = f"Reviewer feedback: {review_response}"
        passed = False
    else:
        feedback = "Acceptable"
        passed = True
        
    return {
        "feedback": feedback,
        "passed_review": passed
    }

async def send_node_async(state: AgentState):
    print("\n--- [NODE: SEND] Connecting to Gmail MCP & sending email ---")
    
    # Define local Gmail MCP Server running via python script
    server_script = os.path.abspath("gmail_mcp_server.py")
    print(f"Launching Gmail MCP server from: {server_script}")
    
    client = MultiServerMCPClient(
        {
            "gmail": {
                "transport": "stdio",
                "command": "python3",
                "args": [server_script],
            }
        }
    )
    tools = await client.get_tools()
    # Find the send_email tool
    send_tool = None
    for t in tools:
        if "send_email" in t.name:
            send_tool = t
            break
    
    if not send_tool:
        raise RuntimeError(f"Could not find send_email tool. Available tools: {[t.name for t in tools]}")
        
    print(f"Calling tool: {send_tool.name}")
    result = await send_tool.ainvoke({
        "to": state["recipient_email"],
        "subject": state["subject"],
        "body": state["body"]
    })
    print(f"Tool execution result: {result}")
        
    return state

# 3. Router
def route_after_review(state: AgentState) -> Literal["send", "draft", "end"]:
    if state.get("passed_review", False):
        return "send"
    elif state.get("attempts", 0) >= 3:
        print("Maximum review attempts reached. Ending workflow without sending.")
        return "end"
    else:
        print(f"Draft rejected. Loop back to draft. Feedback: {state.get('feedback')}")
        return "draft"

# 4. Build workflow
def build_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("draft", draft_node)
    workflow.add_node("review", review_node)
    
    # We wrap the async send node so it works with synchronous graph execution or invoke
    def send_node_sync_wrapper(state: AgentState):
        return asyncio.run(send_node_async(state))
        
    workflow.add_node("send", send_node_sync_wrapper)
    
    workflow.set_entry_point("draft")
    workflow.add_edge("draft", "review")
    
    workflow.add_conditional_edges(
        "review",
        route_after_review,
        {
            "send": "send",
            "draft": "draft",
            "end": END
        }
    )
    workflow.add_edge("send", END)
    
    return workflow.compile()

if __name__ == "__main__":
    import sys
    app = build_workflow()
    
    # Ensure credentials are present or inform the user
    if not os.path.exists("credentials.json") and "GMAIL_CREDENTIALS_PATH" not in os.environ:
        print("Error: credentials.json is missing in current workspace. Please follow the instructions to set up OAuth.", file=sys.stderr)
        sys.exit(1)
        
    if len(sys.argv) > 1:
        recipient = sys.argv[1].strip()
    else:
        recipient = os.environ.get("TEST_RECIPIENT_EMAIL", "").strip()
        
    if not recipient:
        recipient = input("Enter test recipient email: ").strip()
        
    if not recipient:
        print("Recipient email is required.", file=sys.stderr)
        sys.exit(1)
        
    initial_state = {
        "recipient_email": recipient,
        "recipient_name": "Dr. Sarah Jenkins",
        "context": "Seeking a research internship in machine learning and neural networks for the Fall semester, citing interest in her recent paper on transformer efficiency.",
        "subject": "",
        "body": "",
        "feedback": "",
        "passed_review": False,
        "attempts": 0
    }
    
    print("\nRunning Cold Email Agent workflow...")
    final_state = app.invoke(initial_state)
    print("\nWorkflow completed!")
