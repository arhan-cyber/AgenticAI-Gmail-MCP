import sys
from typing import TypedDict, List
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, START, END

# Define the AgentState structure
class AgentState(TypedDict):
    query: str
    current_answer: str
    evaluation_feedback: str
    loop_count: int
    history: List[str]

# 1. Generate Node
def generate_node(state: AgentState):
    print("\n" + "="*40)
    print("--- [NODE: GENERATE] Generating initial draft ---")
    print("="*40)
    
    llm = Ollama(model="qwen2.5:1.5b")
    query = state["query"]
    
    prompt = f"""You are a helpful research assistant. Answer the following query clearly and concisely.

Query: {query}
Answer:"""
    
    initial_answer = llm.invoke(prompt).strip()
    print(f"Initial Answer Draft:\n{initial_answer}")
    
    return {
        "current_answer": initial_answer,
        "loop_count": 0,
        "history": state.get("history", []) + ["Generated initial answer draft"]
    }

# 2. Evaluate Node
def evaluate_node(state: AgentState):
    print("\n" + "="*40)
    print("--- [NODE: EVALUATE] Checking answer quality ---")
    print("="*40)
    
    answer = state.get("current_answer", "")
    loop_count = state.get("loop_count", 0)
    word_count = len(answer.split())
    
    print(f"Word count: {word_count}")
    print(f"Loop count: {loop_count}")
    
    # We must loop at least once before terminating.
    # We also check if the length of the answer is sufficient.
    if loop_count < 1:
        feedback = "The answer is a good start, but needs more detailed expansion, key context, and depth."
        print(f"Result: Rejecting draft (forced first loop). Feedback: {feedback}")
    elif word_count < 80:
        feedback = f"The answer is too short ({word_count} words). Please expand it to at least 80 words."
        print(f"Result: Rejecting draft (too short). Feedback: {feedback}")
    else:
        feedback = "Acceptable"
        print("Result: Acceptable! Terminating the workflow.")
        
    return {
        "evaluation_feedback": feedback,
        "history": state.get("history", []) + [f"Evaluated: loop_count={loop_count}, word_count={word_count}, status={feedback}"]
    }

# 3. Refine Node
def refine_node(state: AgentState):
    print("\n" + "="*40)
    print("--- [NODE: REFINE] Refining and expanding draft ---")
    print("="*40)
    
    llm = Ollama(model="qwen2.5:1.5b")
    query = state["query"]
    current_answer = state["current_answer"]
    feedback = state["evaluation_feedback"]
    
    refine_prompt = f"""You are an expert research editor. 
Your task is to refine, expand, and improve the current answer to the query based on the feedback provided.

Query: {query}
Current Answer: {current_answer}
Feedback for improvement: {feedback}

Please provide a significantly more detailed, comprehensive, and well-structured response (aim for 80-150 words).

Improved Answer:"""
    
    refined_answer = llm.invoke(refine_prompt).strip()
    next_loop_count = state.get("loop_count", 0) + 1
    print(f"Refined Answer (Loop {next_loop_count}):\n{refined_answer}")
    
    return {
        "current_answer": refined_answer,
        "loop_count": next_loop_count,
        "history": state.get("history", []) + [f"Refined answer (loop {next_loop_count})"]
    }

# Conditional routing logic
def route_after_evaluation(state: AgentState):
    feedback = state.get("evaluation_feedback", "")
    if feedback == "Acceptable":
        return "end"
    else:
        return "refine"

def build_workflow():
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("generate", generate_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("refine", refine_node)
    
    # Define execution edges
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "evaluate")
    
    # Add conditional edge from evaluate
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "refine": "refine",
            "end": END
        }
    )
    
    # Loop back from refine to evaluate
    workflow.add_edge("refine", "evaluate")
    
    # Compile graph
    return workflow.compile()

if __name__ == "__main__":
    app = build_workflow()
    
    # Input query
    query = "What is the Model Context Protocol (MCP) and how does it solve integration issues?"
    print(f"Input Query: {query}\n")
    
    initial_state = {
        "query": query,
        "current_answer": "",
        "evaluation_feedback": "",
        "loop_count": 0,
        "history": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    print("\n" + "="*80)
    print("FINAL WORKFLOW STATE:")
    print("="*80)
    import pprint
    pprint.pprint(final_state)
