import os
import time
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Define custom tools
@tool
def calculator(expression: str) -> str:
    """Useful for when you need to answer questions about math/arithmetic expressions.
    Input should be a mathematical expression like '2 + 2' or '3 * 5'."""
    try:
        # A simple, safe evaluation helper
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression."
        # Evaluate safely
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error: Could not evaluate expression. {str(e)}"

# Instantiate the tools
search_tool = DuckDuckGoSearchRun()
tools = [search_tool, calculator]

# 2. Set up the LLM with rate limiting (5 requests per minute, max 20 requests)
class RateLimitedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    _request_timestamps = []
    _total_requests = 0
    _max_total_requests = 20

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if RateLimitedChatGoogleGenerativeAI._total_requests >= RateLimitedChatGoogleGenerativeAI._max_total_requests:
            raise RuntimeError("API rate limit exceeded: Total request cap of 20 reached.")
        
        while True:
            now = time.time()
            # Keep only timestamps within the last 60 seconds
            RateLimitedChatGoogleGenerativeAI._request_timestamps = [
                t for t in RateLimitedChatGoogleGenerativeAI._request_timestamps if now - t < 60
            ]
            if len(RateLimitedChatGoogleGenerativeAI._request_timestamps) < 5:
                break
            
            sleep_time = 60 - (now - RateLimitedChatGoogleGenerativeAI._request_timestamps[0])
            if sleep_time > 0:
                print(f"\n[Rate Limiter] 5 RPM limit reached. Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
        
        RateLimitedChatGoogleGenerativeAI._request_timestamps.append(time.time())
        RateLimitedChatGoogleGenerativeAI._total_requests += 1
        print(f"\n[Rate Limiter] Request {RateLimitedChatGoogleGenerativeAI._total_requests}/{RateLimitedChatGoogleGenerativeAI._max_total_requests} initiated.")
        
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

llm = RateLimitedChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 3. ReAct prompt template
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# 4. Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create the executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 5. Queries to demonstrate
queries = [
    "What is the capital of France? And what is the square root of its population in millions? (Calculate the square root of just the population number in millions, e.g., if population is 2.1 million, calculate sqrt(2.1))",
    "Who directed the movie Interstellar and what is their current age multiplied by 3?",
    "Search for the height of Mount Everest in meters, and then divide that number by 100."
]

if __name__ == "__main__":
    for i, query in enumerate(queries, 1):
        if i > 1:
            print("\nWaiting 2 seconds to avoid rate limits...")
            time.sleep(2)
        print("\n" + "="*80)
        print(f"QUERY {i}: {query}")
        print("="*80)
        try:
            result = agent_executor.invoke({"input": query})
            print("\nResult:")
            print(result["output"])
        except Exception as e:
            print(f"\nError running query: {e}")
