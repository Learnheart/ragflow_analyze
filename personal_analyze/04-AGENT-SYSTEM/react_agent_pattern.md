# ReAct Agent Pattern Implementation

## Tong Quan

ReAct (Reasoning + Acting) la pattern cho phep LLM thuc hien reasoning va goi tools trong vong lap iterative. RAGFlow implement ReAct pattern trong Agent component, ho tro multi-tool orchestration, parallel tool execution, va memory-augmented reasoning.

## File Location
```
/agent/component/agent_with_tools.py
/rag/prompts/generator.py
```

## Architecture

```
                       REACT AGENT PATTERN

                         User Query
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TASK ANALYSIS                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  analyze_task()                                         │   │
│  │  - Understand user intent                               │   │
│  │  - Identify required tools                              │   │
│  │  - Create task breakdown                                │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     REACT LOOP                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  for round in range(max_rounds):                        │   │
│  │      1. next_step() → Decide action(s)                  │   │
│  │      2. Parse tool calls from response                  │   │
│  │      3. Execute tools (parallel if independent)         │   │
│  │      4. reflect() → Analyze results                     │   │
│  │      5. Continue or COMPLETE_TASK                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL RESPONSE                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  - Synthesize all gathered information                  │   │
│  │  - Generate citations if enabled                        │   │
│  │  - Stream response to user                              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Component

### AgentParam Class

```python
class AgentParam(LLMParam, ToolParamBase):
    """
    Agent = LLM + Tools

    Inherits from both LLMParam and ToolParamBase.
    """

    def __init__(self):
        # Tool metadata for supervisor agents
        self.meta: ToolMeta = {
            "name": "agent",
            "description": "This is an agent for a specific task.",
            "parameters": {
                "user_prompt": {
                    "type": "string",
                    "description": "Order to send to the agent.",
                    "required": True
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why this agent is being invoked.",
                    "required": True
                },
                "context": {
                    "type": "string",
                    "description": "Background information for the agent.",
                    "required": True
                }
            }
        }
        super().__init__()

        self.function_name = "agent"
        self.tools = []      # List of tool configurations
        self.mcp = []        # MCP server tools
        self.max_rounds = 5  # Maximum ReAct iterations
        self.description = ""
```

### Agent Initialization

```python
class Agent(LLM, ToolBase):
    component_name = "Agent"

    def __init__(self, canvas, id, param: LLMParam):
        LLM.__init__(self, canvas, id, param)

        # 1. Load configured tools
        self.tools = {}
        for cpn in self._param.tools:
            cpn = self._load_tool_obj(cpn)
            self.tools[cpn.get_meta()["function"]["name"]] = cpn

        # 2. Initialize LLM with tool support
        self.chat_mdl = LLMBundle(
            self._canvas.get_tenant_id(),
            TenantLLMService.llm_id2llm_type(self._param.llm_id),
            self._param.llm_id,
            max_rounds=self._param.max_rounds,
            verbose_tool_use=True
        )

        # 3. Collect tool metadata
        self.tool_meta = [v.get_meta() for _, v in self.tools.items()]

        # 4. Load MCP server tools
        for mcp in self._param.mcp:
            _, mcp_server = MCPServerService.get_by_id(mcp["mcp_id"])
            tool_call_session = MCPToolCallSession(mcp_server, mcp_server.variables)
            for tnm, meta in mcp["tools"].items():
                self.tool_meta.append(mcp_tool_metadata_to_openai_tool(meta))
                self.tools[tnm] = tool_call_session

        # 5. Setup tool call session
        self.callback = partial(self._canvas.tool_use_callback, id)
        self.toolcall_session = LLMToolPluginCallSession(self.tools, self.callback)
```

## ReAct Loop Implementation

```python
def _react_with_tools_streamly(self, prompt, history: list[dict],
                                use_tools, user_defined_prompt={}):
    """
    Main ReAct loop with streaming support.

    Args:
        prompt: System prompt
        history: Conversation history
        use_tools: List to collect tool usage
        user_defined_prompt: Custom prompts for reflection, etc.

    Yields:
        (delta_answer, token_count) tuples
    """
    token_count = 0
    tool_metas = self.tool_meta
    hist = deepcopy(history)

    # Optimize multi-turn conversations
    if len(hist) > 3:
        user_request = full_question(messages=history, chat_mdl=self.chat_mdl)
        self.callback("Multi-turn conversation optimization", {}, user_request)
    else:
        user_request = history[-1]["content"]

    # Phase 1: Task Analysis
    task_desc = analyze_task(
        self.chat_mdl, prompt, user_request,
        tool_metas, user_defined_prompt
    )
    self.callback("analyze_task", {}, task_desc)

    # Phase 2: ReAct Loop
    for _ in range(self._param.max_rounds + 1):
        if self.check_if_canceled("Agent streaming"):
            return

        # Step 1: Decide next action(s)
        response, tk = next_step(
            self.chat_mdl, hist, tool_metas,
            task_desc, user_defined_prompt
        )
        token_count += tk
        hist.append({"role": "assistant", "content": response})

        # Step 2: Parse tool calls
        try:
            functions = json_repair.loads(re.sub(r"```.*", "", response))
            if not isinstance(functions, list):
                raise TypeError(f"Expected list, got: {functions}")

            # Step 3: Execute tools (parallel)
            with ThreadPoolExecutor(max_workers=5) as executor:
                threads = []
                for func in functions:
                    name = func["name"]
                    args = func["arguments"]

                    # Check for completion signal
                    if name == COMPLETE_TASK:
                        append_user_content(hist,
                            f"Respond with a formal answer. FORGET about `{COMPLETE_TASK}`.")
                        for txt, tkcnt in self._complete(hist):
                            yield txt, tkcnt
                        return

                    threads.append(executor.submit(self._use_tool, name, args, use_tools))

                # Step 4: Reflect on results
                tool_results = [th.result() for th in threads]
                reflection = reflect(
                    self.chat_mdl, hist, tool_results, user_defined_prompt
                )
                append_user_content(hist, reflection)
                self.callback("reflection", {}, str(reflection))

        except Exception as e:
            # Handle JSON parsing errors
            error = f"\nTool call error: {e}"
            append_user_content(hist, error)

    # Max rounds exceeded - generate final response
    self._handle_max_rounds_exceeded(hist, user_request)
    for txt, tkcnt in self._complete(hist):
        yield txt, tkcnt
```

## Tool Execution

```python
def _use_tool(self, name, args, use_tools):
    """
    Execute a single tool and record usage.
    """
    tool_response = self.toolcall_session.tool_call(name, args)

    use_tools.append({
        "name": name,
        "arguments": args,
        "results": tool_response
    })

    return name, tool_response
```

## Prompt Generation Functions

### Task Analysis

```python
# In /rag/prompts/generator.py

def analyze_task(chat_mdl, sys_prompt, user_request, tools, user_defined_prompt={}):
    """
    Analyze user request and create task description.

    Returns structured task breakdown.
    """
    prompt = f"""
{user_defined_prompt.get('task_analysis', '')}

## Available Tools
{json.dumps([t['function'] for t in tools], indent=2)}

## User Request
{user_request}

Analyze this request and identify:
1. Main goal
2. Required information
3. Suggested tool usage sequence
"""
    return chat_mdl.chat(sys_prompt, [{"role": "user", "content": prompt}], {})
```

### Next Step Decision

```python
COMPLETE_TASK = "__complete_task__"

def next_step(chat_mdl, history, tools, task_desc, user_defined_prompt={}):
    """
    Decide next action(s) based on current state.

    Returns JSON array of tool calls:
    [
        {"name": "tool_name", "arguments": {...}},
        {"name": "__complete_task__", "arguments": {}}
    ]
    """
    tools_desc = json.dumps([t['function'] for t in tools], indent=2)

    prompt = f"""
{user_defined_prompt.get('plan_generation', '')}

## Task Description
{task_desc}

## Available Tools
{tools_desc}

## Special Action
When the task is complete, return:
{{"name": "{COMPLETE_TASK}", "arguments": {{}}}}

Based on the conversation, decide the next action(s).
Return a JSON array of tool calls.
"""
    response = chat_mdl.chat("", [*history, {"role": "user", "content": prompt}], {})
    tokens = chat_mdl.token_count

    return response, tokens
```

### Reflection

```python
def reflect(chat_mdl, history, tool_results, user_defined_prompt={}):
    """
    Analyze tool results and plan next steps.

    Args:
        tool_results: List of (tool_name, result) tuples
    """
    results_desc = "\n".join([
        f"**{name}**: {result}"
        for name, result in tool_results
    ])

    prompt = f"""
{user_defined_prompt.get('reflection', '')}

## Tool Execution Results
{results_desc}

Analyze these results:
1. Did the tools provide useful information?
2. Is more information needed?
3. Can we answer the user's question now?

Provide your analysis and suggest next steps.
"""
    return chat_mdl.chat("", [*history, {"role": "user", "content": prompt}], {})
```

## Multi-Agent Support

```python
# Agent as a tool for supervisor agents
def get_meta(self) -> dict[str, Any]:
    """
    Return tool metadata for use by supervisor agents.
    """
    self._param.function_name = self._id.split("-->")[-1]
    m = super().get_meta()

    if hasattr(self._param, "user_prompt") and self._param.user_prompt:
        m["function"]["parameters"]["properties"]["user_prompt"] = self._param.user_prompt

    return m
```

## Tool Types

### Built-in Tools

```python
# Available in /agent/tools/
tools = {
    "Retrieval": "Knowledge base search",
    "Google": "Web search",
    "ExeSQL": "SQL execution",
    "CodeExec": "Python/JS execution",
    "Wikipedia": "Wikipedia search",
    "ArXiv": "Academic paper search",
    "PubMed": "Biomedical literature",
    "Tavily": "Structured web search",
    "BaiduSearch": "Baidu search engine",
    "DuckDuckGo": "DuckDuckGo search",
    "QWeather": "Weather API",
    "YahooFinance": "Stock data",
}
```

### MCP Tools

```python
# Model Context Protocol tools
for mcp in self._param.mcp:
    _, mcp_server = MCPServerService.get_by_id(mcp["mcp_id"])
    tool_call_session = MCPToolCallSession(mcp_server, mcp_server.variables)

    for tnm, meta in mcp["tools"].items():
        self.tool_meta.append(mcp_tool_metadata_to_openai_tool(meta))
        self.tools[tnm] = tool_call_session
```

## Memory System

```python
def get_useful_memory(self, goal: str, sub_goal: str, topn=3,
                       user_defined_prompt: dict = {}) -> str:
    """
    Retrieve relevant memories for current task.

    Uses LLM to rank memories by relevance.
    """
    mems = self._canvas.get_memory()

    # Rank memories by relevance
    rank = rank_memories(
        self.chat_mdl, goal, sub_goal,
        [summ for (user, assist, summ) in mems],
        user_defined_prompt
    )

    try:
        rank = json_repair.loads(re.sub(r"```.*", "", rank))[:topn]
        mems = [mems[r] for r in rank]
        return "\n\n".join([
            f"User: {u}\nAgent: {a}"
            for u, a, _ in mems
        ])
    except Exception as e:
        logging.exception(e)
        return "Error occurred."

def add_memory(self, user: str, assist: str, func_name: str,
               params: dict, results: str, user_defined_prompt: dict = {}):
    """
    Summarize tool call and add to memory.
    """
    summ = tool_call_summary(
        self.chat_mdl, func_name, params, results, user_defined_prompt
    )
    logging.info(f"[MEMORY]: {summ}")
    self._canvas.add_memory(user, assist, summ)
```

## Error Handling

```python
def _handle_max_rounds_exceeded(self, hist, user_request):
    """
    Handle case when max rounds are exceeded.
    """
    logging.warning(f"Exceed max rounds: {self._param.max_rounds}")

    final_instruction = f"""
{user_request}

IMPORTANT: You have reached the conversation limit.
Based on ALL the information gathered so far:
1. SYNTHESIZE all information collected
2. Provide a COMPLETE response using existing data
3. Structure your response as a FINAL DELIVERABLE
4. DO NOT mention conversation limits
5. Focus on delivering VALUE with available data

Respond immediately with your final comprehensive answer.
"""
    append_user_content(hist, final_instruction)
```

## Output Structure

```python
# Agent outputs
outputs = {
    "content": "Final response text",
    "use_tools": [
        {
            "name": "Retrieval",
            "arguments": {"query": "..."},
            "results": "..."
        },
        {
            "name": "Google",
            "arguments": {"query": "..."},
            "results": "..."
        }
    ],
    "_created_time": 1234567890.123,
    "_elapsed_time": 15.5
}
```

## DSL Configuration

```json
{
    "Agent:Research": {
        "obj": {
            "component_name": "Agent",
            "params": {
                "llm_id": "gpt-4o@OpenAI",
                "sys_prompt": "You are a research assistant.",
                "prompts": [{"role": "user", "content": "{{sys.query}}"}],
                "max_rounds": 5,
                "tools": [
                    {
                        "component_name": "Retrieval",
                        "params": {"kb_ids": ["kb_123"]}
                    },
                    {
                        "component_name": "Google",
                        "params": {}
                    }
                ],
                "mcp": [
                    {
                        "mcp_id": "mcp_server_123",
                        "tools": {"custom_tool": {...}}
                    }
                ]
            }
        },
        "downstream": ["Message:Output"],
        "upstream": ["begin"]
    }
}
```

## Related Files

- `/agent/component/agent_with_tools.py` - Agent component
- `/agent/tools/base.py` - Tool base classes
- `/rag/prompts/generator.py` - ReAct prompts
- `/common/mcp_tool_call_conn.py` - MCP integration
