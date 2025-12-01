# Variable System - Interpolation va State Management

## Tong Quan

Variable System trong RAGFlow Agent cho phep truyen du lieu giua cac components thong qua variable references. He thong nay ho tro system variables, component outputs, nested property access, va environment variables.

## File Location
```
/agent/canvas.py          # Variable resolution
/agent/component/base.py  # Variable reference pattern
```

## Architecture

```
                     VARIABLE SYSTEM ARCHITECTURE

┌─────────────────────────────────────────────────────────────────┐
│                      VARIABLE TYPES                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │  System Vars     │  │  Component Refs   │  │  Env Vars    │ │
│  │                  │  │                   │  │              │ │
│  │  sys.query       │  │  cpn_id@output    │  │  env.API_KEY │ │
│  │  sys.user_id     │  │  LLM:0@content    │  │  env.DEBUG   │ │
│  │  sys.files       │  │  retrieval@chunks │  │              │ │
│  │  sys.conv_turns  │  │                   │  │              │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
│                                                                  │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                   VARIABLE RESOLUTION                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Pattern: {{component_id@output_name.property.path}}    │   │
│  │                                                          │   │
│  │  Examples:                                               │   │
│  │  - {{sys.query}}              → User input               │   │
│  │  - {{LLM:0@content}}          → LLM response             │   │
│  │  - {{retrieval@chunks[0]}}    → First chunk              │   │
│  │  - {{agent@result.items.name}}→ Nested property          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Variable Reference Pattern

```python
# In ComponentBase
variable_ref_patt = r"\{* *\{([a-zA-Z:0-9]+@[A-Za-z0-9_.]+|sys\.[A-Za-z0-9_.]+|env\.[A-Za-z0-9_.]+)\} *\}*"

# Matches:
# {{component_id@variable_name}}
# {{sys.variable_name}}
# {{env.VARIABLE_NAME}}
# {component_id@variable_name}  (single braces also work)
```

## System Variables (globals)

```python
class Canvas(Graph):
    def __init__(self, dsl: str, tenant_id=None, task_id=None):
        self.globals = {
            "sys.query": "",           # Current user query
            "sys.user_id": tenant_id,  # Tenant/user ID
            "sys.conversation_turns": 0,  # Turn count
            "sys.files": []            # Uploaded files (base64)
        }
```

| Variable | Type | Description |
|----------|------|-------------|
| `sys.query` | string | Current user input/question |
| `sys.user_id` | string | Tenant ID for multi-tenancy |
| `sys.conversation_turns` | int | Number of conversation turns |
| `sys.files` | list | Uploaded files as base64 strings |

## Component Output References

```python
# Format: {{component_id@output_name}}

# Examples:
{{begin@prologue}}           # Begin component's prologue
{{LLM:Planning@content}}     # LLM component's content output
{{retrieval_0@chunks}}       # Retrieval component's chunks
{{retrieval_0@formalized_content}}  # Formatted retrieval results
{{categorize_0@category_name}}      # Selected category
{{agent_0@use_tools}}        # Tools used by agent
```

## Variable Resolution Flow

```python
def get_variable_value(self, exp: str) -> Any:
    """
    Resolve variable expression to actual value.

    Args:
        exp: Variable expression like "sys.query" or "LLM:0@content"

    Returns:
        Resolved value (string, dict, list, etc.)
    """
    # Clean up expression
    exp = exp.strip("{").strip("}").strip(" ").strip("{").strip("}")

    # System/global variables (no @)
    if exp.find("@") < 0:
        return self.globals[exp]

    # Component references (with @)
    cpn_id, var_nm = exp.split("@")
    cpn = self.get_component(cpn_id)

    if not cpn:
        raise Exception(f"Can't find variable: '{cpn_id}@{var_nm}'")

    # Handle nested property access
    parts = var_nm.split(".", 1)
    root_key = parts[0]
    rest = parts[1] if len(parts) > 1 else ""

    root_val = cpn["obj"].output(root_key)

    if not rest:
        return root_val

    return self.get_variable_param_value(root_val, rest)
```

## Nested Property Access

```python
def get_variable_param_value(self, obj: Any, path: str) -> Any:
    """
    Navigate nested properties using dot notation.

    Supports:
    - Dict access: obj.key
    - List access: obj.0 or obj[0]
    - Nested paths: obj.key1.key2.0

    Examples:
        {{agent@result.items[0].name}}
        {{data@response.data.users.0.email}}
    """
    cur = obj
    if not path:
        return cur

    for key in path.split('.'):
        if cur is None:
            return None

        # Auto-parse JSON strings
        if isinstance(cur, str):
            try:
                cur = json.loads(cur)
            except Exception:
                return None

        # Dict access
        if isinstance(cur, dict):
            cur = cur.get(key)
            continue

        # List/tuple access
        if isinstance(cur, (list, tuple)):
            try:
                idx = int(key)
                cur = cur[idx]
            except Exception:
                return None
            continue

        # Object attribute access
        cur = getattr(cur, key, None)

    return cur
```

## Variable Setting

```python
def set_variable_value(self, exp: str, value):
    """
    Set variable value in canvas state.
    """
    exp = exp.strip("{").strip("}").strip(" ")

    # System variables
    if exp.find("@") < 0:
        self.globals[exp] = value
        return

    # Component outputs
    cpn_id, var_nm = exp.split("@")
    cpn = self.get_component(cpn_id)

    if not cpn:
        raise Exception(f"Can't find variable: '{cpn_id}@{var_nm}'")

    parts = var_nm.split(".", 1)
    root_key = parts[0]
    rest = parts[1] if len(parts) > 1 else ""

    if not rest:
        cpn["obj"].set_output(root_key, value)
        return

    # Nested property setting
    root_val = cpn["obj"].output(root_key) or {}
    cpn["obj"].set_output(
        root_key,
        self.set_variable_param_value(root_val, rest, value)
    )
```

## String Interpolation

```python
def get_value_with_variable(self, value: str) -> Any:
    """
    Replace all {{variable}} references in a string.

    Handles:
    - Multiple variables in one string
    - Mixed static and dynamic content
    - Generator/partial function results
    """
    pat = re.compile(
        r"\{* *\{([a-zA-Z:0-9]+@[A-Za-z0-9_.]+|sys\.[A-Za-z0-9_.]+|env\.[A-Za-z0-9_.]+)\} *\}*"
    )
    out_parts = []
    last = 0

    for m in pat.finditer(value):
        out_parts.append(value[last:m.start()])
        key = m.group(1)
        v = self.get_variable_value(key)

        if v is None:
            rep = ""
        elif isinstance(v, partial):
            # Handle streaming generators
            buf = []
            for chunk in v():
                buf.append(chunk)
            rep = "".join(buf)
        elif isinstance(v, str):
            rep = v
        else:
            rep = json.dumps(v, ensure_ascii=False)

        out_parts.append(rep)
        last = m.end()

    out_parts.append(value[last:])
    return "".join(out_parts)
```

## Component Input Resolution

```python
# In ComponentBase
def get_input(self, key: str = None) -> Union[Any, dict[str, Any]]:
    """
    Get input values with variable interpolation.
    """
    res = {}
    for var, o in self.get_input_elements().items():
        v = self.get_param(var)
        if v is None:
            continue

        # Check if variable reference
        if isinstance(v, str) and self._canvas.is_reff(v):
            self.set_input_value(var, self._canvas.get_variable_value(v))
        else:
            self.set_input_value(var, v)

        res[var] = self.get_input_value(var)

    if key:
        return res.get(key)
    return res

def get_input_elements_from_text(self, txt: str) -> dict[str, dict[str, str]]:
    """
    Extract all {{variable}} references from text.

    Returns dict with variable info:
    {
        "LLM:0@content": {
            "name": "LLM:0@content",
            "value": "actual value",
            "_retrival": retrieval_data,
            "_cpn_id": "LLM:0"
        }
    }
    """
    res = {}
    for r in re.finditer(self.variable_ref_patt, txt, flags=re.IGNORECASE|re.DOTALL):
        exp = r.group(1)
        cpn_id, var_nm = exp.split("@") if exp.find("@") > 0 else ("", exp)
        res[exp] = {
            "name": (self._canvas.get_component_name(cpn_id) + f"@{var_nm}") if cpn_id else exp,
            "value": self._canvas.get_variable_value(exp),
            "_retrival": self._canvas.get_variable_value(f"{cpn_id}@_references") if cpn_id else None,
            "_cpn_id": cpn_id
        }
    return res
```

## State Management

### Canvas State Structure

```python
# Canvas DSL state
{
    "components": {
        "begin": {
            "obj": {...},
            "downstream": [...],
            "upstream": [...]
        }
    },
    "globals": {
        "sys.query": "User question",
        "sys.user_id": "tenant_123",
        "sys.conversation_turns": 5,
        "sys.files": []
    },
    "history": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "memory": [
        ["user_msg", "assist_msg", "summary"]
    ],
    "retrieval": {
        "chunks": {...},
        "doc_aggs": {...}
    },
    "path": ["begin", "LLM:0", "Message:0"]
}
```

### History Management

```python
def get_history(self, window_size: int) -> list[dict]:
    """
    Get conversation history with size limit.
    """
    return self.history[-window_size:] if window_size > 0 else self.history

def add_history(self, role: str, content: str):
    """
    Add message to history.
    """
    self.history.append({"role": role, "content": content})
```

### Memory Management

```python
def get_memory(self) -> list:
    """
    Get agent memory (tool call summaries).
    """
    return self.memory

def add_memory(self, user: str, assist: str, summary: str):
    """
    Add tool call summary to memory.
    """
    self.memory.append([user, assist, summary])
```

### Reference Tracking

```python
def get_reference(self) -> dict:
    """
    Get retrieval references.

    Returns:
        {
            "chunks": {chunk_id: chunk_data},
            "doc_aggs": {doc_id: aggregation_data}
        }
    """
    return self.retrieval

def set_reference(self, key: str, value: Any):
    """
    Set retrieval reference.
    """
    if key == "chunks":
        self.retrieval["chunks"].update(value)
    elif key == "doc_aggs":
        self.retrieval["doc_aggs"].update(value)
```

## Usage Examples

### In Prompts

```json
{
    "sys_prompt": "You are helping user: {{sys.user_id}}",
    "prompts": [
        {
            "role": "user",
            "content": "Based on this context:\n{{retrieval_0@formalized_content}}\n\nAnswer: {{sys.query}}"
        }
    ]
}
```

### In Message Component

```json
{
    "content": "Here's what I found:\n\n{{LLM:Planning@content}}\n\nSources: {{retrieval_0@doc_aggs}}"
}
```

### In Conditional Routing

```json
{
    "conditions": [
        {
            "var": "{{categorize@category_name}}",
            "op": "eq",
            "value": "technical"
        }
    ]
}
```

### In Iteration

```json
{
    "items_ref": "{{data_processor@items}}",
    "item_var": "current_item"
}
```

## Variable Assigner Component

```python
class VariableAssigner(ComponentBase):
    """
    Assign values to variables dynamically.

    Used for:
    - Setting computed values
    - Transforming data
    - Cross-component data flow
    """

    def _invoke(self, **kwargs):
        for assignment in self._param.assignments:
            target = assignment["target"]  # Variable to set
            source = assignment["source"]  # Value or expression

            # Resolve source if it's a variable reference
            if self._canvas.is_reff(source):
                value = self._canvas.get_variable_value(source)
            else:
                value = source

            # Set the target variable
            self._canvas.set_variable_value(target, value)
```

## Related Files

- `/agent/canvas.py` - Canvas class with variable resolution
- `/agent/component/base.py` - ComponentBase with reference pattern
- `/agent/component/variable_assigner.py` - Variable assignment
- `/agent/component/variable_aggregator.py` - Variable aggregation
