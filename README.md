# MCP, LangChain and LangGraph

## **What is an AI Agent?**

In Weeks 1 and 2, we learned how to make LLMs remarkably useful: crafting prompts that produce exactly the right output, and building RAG pipelines that let models answer from our own documents. That was one paradigm: we ask, the model answers, the conversation ends.

This week introduces a fundamentally different paradigm. An AI agent does not just respond, it reasons about a goal, decides which tools to use, executes those tools, observes the results, and decides what to do next. The loop continues until the goal is achieved (or failed). The model is no longer a responder; it is a decision-maker embedded inside a workflow.

Every true AI agent whether built with LangChain, LangGraph, or raw API calls shares these four defining capabilities:

- **Perception:** The ability to receive inputs beyond the user's message. Agents can read documents, call APIs, query databases, and browse the web.
- **Planning:** The ability to decompose a complex goal into a sequence of sub-tasks and decide an order of operations.
- **Action:** The ability to execute tools: code, API calls, file operations that change the state of the world.
- **Memory:** The ability to remember what happened in previous steps and carry context forward so each action is informed by history.

**CRITICAL INSIGHT:** The LLM is the brain and it does the reasoning. Tools are the hands and they do the work. Memory is the whiteboard and it holds the intermediate state. The agent framework (LangChain/LangGraph/MCP) is the nervous system and it connects everything together and manages the flow of information

## Model Context Protocol (MCP)

Model Context Protocol (MCP) Explained for Beginners: AI Flight Booking Demo!

Watch this video to get an understanding of AI Agents and MCP.

Note: This video contains a free lab which will deepen your understanding of the topic. Make sure to follow along with the instructions provided in the video.

### **Intuitive Introduction: The Integration Problem**

Before MCP existed, every AI application that wanted to connect to an external tool had to write custom integration code for that specific tool with that specific model. If you used GPT-4 and wanted it to query your database, you wrote a GPT-4 specific database integration. If you then switched to Claude, you had to rewrite it. If you added Slack integration, you wrote another custom connector. If you had five models and ten tools, you potentially needed fifty different integration implementations.

This is known as the N × M integration problem. N models × M tools = N×M custom connectors. This is not sustainable.

MCP solves this by introducing a universal standard, a common language that all models and all tools agree to speak. Once a tool exposes an MCP server, any MCP-compatible model can use it. Once a model supports MCP, it can use any MCP server. The N×M problem collapses to N+M.

!Screenshot 2026-06-22 at 11.58.43 PM.png

### **Core Concepts: What MCP Actually is?**

MCP (Model Context Protocol) was introduced by Anthropic in November 2024 and immediately open-sourced to encourage industry-wide adoption. It is a specification, a set of rules for how AI applications and external tools exchange information. It is not a library or a framework per se; it is a protocol, like HTTP is a protocol for web communication.

!Screenshot 2026-06-23 at 12.04.23 AM.png

**What MCP Servers Expose:**

Every MCP server exposes up to three types of capabilities:

- **Tools:** Functions the model can call (search_web, create_issue, query_database). The model sees the tool name, description, and parameter schema, and decides when to call it.
- **Resources:** Data sources the host can read (a file, a database row, a webpage). Unlike tools, resources are read-only and do not perform actions.
- **Prompts:** Reusable prompt templates the server makes available to the host application. These are user-controlled, not model-controlled.

**The Restaurant Analogy:** The MCP Host is the restaurant management system. The MCP Client is the waiter who takes your order to the kitchen. The MCP Server is the kitchen that prepares your specific dish (the tool result). You (the user) interact with the management system; you never talk directly to the kitchen

### **Deep Dive: How MCP Works Internally**

MCP is built on top of JSON-RPC, which is a lightweight remote procedure call protocol that encodes function calls and responses as JSON objects. The communication is stateful,  a session is established between client and server, and context is maintained across multiple calls within that session.

**The MCP Communication Flow**

1. The host application starts and discovers which MCP servers are configured (via a config file, typically mcp.json).
2. For each server, the host spawns an MCP client and establishes a connection (via stdio for local processes, or HTTP/SSE for remote servers).
3. The client calls the server's 'list_tools' endpoint to discover all available tools, their descriptions, and their parameter schemas.
4. This tool list is injected into the LLM's context so the model knows what capabilities are available.
5. When the user asks a question, the LLM reasons about which tool(s) to call and generates a structured tool-call request.
6. The MCP client receives the tool-call request, forwards it to the appropriate MCP server.
7. The server executes the action (queries the database, calls the API, reads the file) and returns the result.
8. The result flows back to the LLM, which incorporates it into its reasoning and either calls another tool or produces a final response.

**MCP Request-Response Flow**

!Screenshot 2026-06-23 at 12.15.31 PM.png

**Transport Mechanisms:** 

MCP supports multiple transport layers depending on where the server runs:

!Screenshot 2026-06-24 at 1.02.32 AM.png

## LangChain

[LangChain Mastery in 2025 | Full 5 Hour Course [LangChain v0.3]](https://youtu.be/Cyv-dgv80kE?si=xqa7GIuT80QYoA07)

This is a comprehensive 5-hour video lecture covering LangChain v0.3 in depth. You do not need to watch it all in one sitting. Use the written material provided to understand the concepts in depth.

### **Intuitive Introduction: Why We Need a Framework**

You can build an AI agent using raw LLM API calls, no framework needed. You send a prompt, get a response, parse it, call a tool based on it, format the result, send another prompt. It works. And for simple applications, it is probably the right approach.

But consider what happens as complexity grows. You want to add memory so the agent remembers the last ten messages. You want to support multiple LLM providers without rewriting your code. You want to chain multiple steps, retrieve documents, then summarize, then translate in a clean pipeline. You want to observe and debug what the agent is doing inside. You want to add streaming so users see partial results instead of waiting.

Every one of these features requires significant engineering work when built from scratch. And every team building AI applications faces the same set of problems. LangChain exists to solve these problems once, correctly, for everyone.

**What LangChain is:** LangChain is a high-level Python (and JavaScript) framework for building applications powered by language models. It provides standardized abstractions for models, prompts, memory, tools, and chains so you can focus on your application logic rather than plumbing.

### **Core Concepts: The Building Blocks of LangChain**

!Screenshot 2026-06-24 at 1.05.31 AM.png

**Building Block 1: Models**

LangChain provides a unified interface for interacting with any LLM: OpenAI's GPT series, Anthropic's Claude, Google's Gemini, open-source models via HuggingFace or Ollama. You swap models by changing one line of code. Your application logic stays exactly the same.

!Screenshot 2026-06-24 at 1.06.51 AM.png

**Building Block 2: Prompt Templates**

Rather than building prompts by concatenating strings (which is fragile and messy), LangChain provides PromptTemplate which is a structured way to define prompts with named variables that get filled in at runtime.

!Screenshot 2026-06-24 at 1.07.38 AM.png

**Building Block 3: Chains (LCEL: LangChain Expression Language)**

A chain is a sequence of steps where the output of one step becomes the input to the next. LangChain's Expression Language (LCEL) lets you compose these steps using the pipe operator (|), making the data flow visually obvious.

!Screenshot 2026-06-24 at 1.08.20 AM.png

**Building Block 4: Memory**

By default, each LLM call is stateless i.e. the model has no recollection of previous turns in the conversation. LangChain's memory modules solve this by storing conversation history and injecting it into each new prompt.

!Screenshot 2026-06-24 at 1.09.26 AM.png

LangChain supports several memory strategies, each with different tradeoffs:

!Screenshot 2026-06-24 at 1.10.01 AM.png

**Building Block 5: Tools**

A tool in LangChain is any Python function that an agent can call. LangChain provides hundreds of built-in tools (web search, Wikipedia, Python REPL, shell) and lets you easily define your own using the @tool decorator.

!Screenshot 2026-06-24 at 1.10.55 AM.png

**Note:** The docstring of a LangChain tool is not just documentation. It is the description the LLM sees when deciding whether to call the tool. Write it to answer the question: 'When should the agent use this?' A vague docstring leads to the agent using tools at the wrong time or ignoring them entirely.

**Building Block 6: Agents**

A LangChain agent combines a model, a set of tools, and a reasoning strategy. The agent iterates: observe the goal, reason about which tool to use, call the tool, observe the result, reason again, repeat until done.

!Screenshot 2026-06-24 at 1.11.46 AM.png

### **Deep Dive: The ReAct Pattern**

The most common agent reasoning strategy in LangChain is ReAct, which stands for Reason + Act. Understanding ReAct is critical because it is the foundation of nearly all practical agent implementations.

ReAct works by giving the LLM a structured format for its internal reasoning. At each step, the model is asked to produce:

- Thought: The model's reasoning about what to do next. ('I need to find the current stock price of TSLA. I have a get_stock_price tool that can do this.')
- Action: The tool to call and its input. ('get_stock_price("TSLA")')
- Observation: The tool's result, injected back into the prompt.

This cycle repeats: Thought → Action → Observation → Thought → Action → Observation until the model decides it has enough information to produce a Final Answer.

!Screenshot 2026-06-24 at 1.12.39 AM.png

### **LangSmith: Observability for LLM Applications**

When an agent makes a wrong decision, how do you debug it? You cannot add a print statement in the middle of the LLM's reasoning. This is where LangSmith comes in. It is LangChain's observability platform that records every step of every chain and agent run, making it possible to trace exactly what the model saw, what it decided, and why.

- Trace every LLM call: see the exact prompt sent, the response received, and the latency.
- Visualize chain execution: see which nodes ran, in what order, with what inputs and outputs.
- Debug agent loops: see every Thought/Action/Observation cycle.
- Monitor costs: track token usage and API costs per run.

## LangGraph

LangGraph Crash Course For Beginners 2025 | Full 8 Hour Course | LangGraph 0.4V LATEST!

This video explains LangGraph from the ground up, showing how it extends LangChain with graph-based workflows. It is shorter and more focused than the LangChain course.

### **Intuitive Introduction: The Problem LangChain Cannot Solve**

LangChain excels at linear workflows: step A, then step B, then step C. But real agent behavior is rarely linear. Consider what a skilled human analyst does when researching a topic: they search, find something interesting, search more specifically, realize they need data from a different source, loop back, compare findings, ask a clarifying question, revise their hypothesis, and search again. Their workflow is a graph with loops, branches, and conditional paths, not a chain.

LangChain's standard chains flow in one direction only. If an agent using a LangChain chain needs to loop back and search again after seeing an inadequate result, you have to implement that loop yourself, outside the framework. For complex agents, this leads to messy, hard-to-maintain code.

LangGraph was built by the LangChain team specifically to solve this problem. It models agent workflows as directed graphs where nodes are actions and edges are transitions and makes cycles, conditional branching, and persistent shared state first-class primitives.

**Note:** LangGraph is to LangChain what a flowchart is to a checklist. Checklists (chains) are great for predictable, sequential tasks. Flowcharts (graphs) are great for workflows that need decisions, loops, and branching paths. LangGraph brings graph-based reasoning to AI agents.

### **Core Concepts: The Four Primitives of LangGraph**

**Primitive 1: State**

State is the central, shared memory object that flows through every node in the graph. Think of it as a whiteboard in a conference room: every participant (node) can read from it and write to it, and it captures the current status of the entire workflow.

You define the state schema as a Python TypedDict, a structured dictionary with typed fields. This makes your agent's memory explicit, inspectable, and debuggable.

!Screenshot 2026-06-24 at 1.18.02 AM.png

**Primitive 2: Nodes**

A node is a single unit of computation in the workflow. It is simply a Python function that takes the current state as input and returns an updated state. Nodes can: call an LLM, execute a tool, process data, make a decision. Anything a Python function can do.

!Screenshot 2026-06-24 at 1.18.46 AM.png

**Primitive 3: Edges**

Edges define how control flows between nodes. LangGraph supports two types:

- Normal Edges: Unconditional transitions. After node A finishes, always go to node B.
- Conditional Edges: Transitions that depend on the current state. After node A, look at the state and decide whether to go to node B, node C, or END. This is how loops and branching are implemented.

!Screenshot 2026-06-24 at 1.19.35 AM.png

**Primitive 4: StateGraph**

The StateGraph is the container that holds all nodes and edges together. You assemble the graph by adding nodes, defining the entry point, and adding edges between nodes. Then you compile it into an executable.

!Screenshot 2026-06-24 at 1.21.31 AM.png

### **Deep Dive: How LangGraph Works Internally**

**The Execution Model**

When you call app.invoke(initial_state), LangGraph begins executing at the entry point node. Each node receives the full current state, does its work, and returns a dict of state updates. LangGraph merges these updates into the state (using the reducer functions defined in the TypedDict annotations), then evaluates the outgoing edges to determine which node runs next.

This continues until execution reaches the END node, at which point the final state is returned. The entire execution history, every intermediate state at every node is preserved, making debugging straightforward.

**Persistence and Checkpointing**

One of LangGraph's most powerful features for production systems is checkpointing, the ability to save the state after every node execution and resume from any saved checkpoint. This enables three critical capabilities:

- Human-in-the-loop: Pause the agent at a sensitive decision point, ask a human to review, then resume. This is how production AI systems handle high-stakes actions (deleting data, sending emails, making purchases) safely.
- Fault tolerance: If the execution crashes halfway through (network error, API timeout), resume from the last checkpoint rather than restarting from scratch.
- Long-running workflows: Persist state across sessions. An agent can work on a task for days, stopping and resuming without losing progress.

**Streaming**

LangGraph supports streaming, which lets your application display partial results as each node completes rather than waiting for the full workflow to finish. For agents that might run for 30 seconds or more, streaming is critical for user experience.

### **LangChain vs LangGraph: Knowing When to Use Which**

!Screenshot 2026-06-24 at 1.22.21 AM.png

## **The Complete Picture: MCP + LangChain + LangGraph**

In a typical production system: LangGraph defines the overall agent workflow (the graph of nodes and edges). Inside each LangGraph node, LangChain components do the work (model calls, memory retrieval, output parsing). Tools exposed to the agent are discovered and called via MCP servers, giving the agent access to external data and services.

**A Full Worked Example: The Intelligent Research & Report Agent**

Let us trace through a complete example to see all three layers working together.

Scenario: A user asks: 'Write me a research brief on the current state of quantum computing, with key players, recent breakthroughs, and investment landscape. Save it to my Google Drive.'

**Layer 1: MCP - Tool Discovery**

The agent discovers the following tools via MCP servers configured in its environment:

- web_search (from a web search MCP server)
- arxiv_search (from an arXiv MCP server)
- gdrive_create_doc (from a Google Drive MCP server)
- gdrive_write (from the same Google Drive MCP server)

**Layer 2: LangChain - Components**

The agent uses LangChain's ChatOpenAI model with these tools bound to it. A ConversationSummaryMemory keeps track of what has been researched so far. A PromptTemplate instructs the model to structure its research systematically.

**Layer 3: LangGraph - Workflow**

The LangGraph workflow looks like this:

1. [start] → [plan_research]: LLM generates a research plan with 5 sub-questions.
2. [plan_research] → [search_loop]: For each sub-question, call web_search and arxiv_search.
3. [search_loop] → [evaluate_coverage]: LLM evaluates whether enough information has been gathered. Conditional edge: if insufficient → loop back to [search_loop] with refined queries; if sufficient → proceed.
4. [evaluate_coverage] → [write_report]: LLM synthesizes all gathered information into a structured report.
5. [write_report] → [save_to_drive]: Call gdrive_create_doc and gdrive_write via MCP to save the report.
6. [save_to_drive] → [END]: Return confirmation to the user.
