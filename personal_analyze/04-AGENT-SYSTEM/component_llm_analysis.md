# LLM Component Deep Analysis

## Tong Quan

LLM Component la component quan trong nhat trong Agent System, chiu trach nhiem goi cac Language Models de sinh noi dung. No ho tro streaming output, structured output, image processing, va tich hop voi retrieval context.

## File Location
```
/agent/component/llm.py
```

## Architecture

```
                      LLM COMPONENT ARCHITECTURE

                         User Query
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LLM COMPONENT                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LLMParam                                               │   │
│  │  - llm_id: Model identifier                             │   │
│  │  - sys_prompt: System prompt                            │   │
│  │  - prompts: User prompts template                       │   │
│  │  - temperature, top_p, max_tokens: Generation config    │   │
│  │  - output_structure: JSON schema for structured output  │   │
│  │  - cite: Enable citation generation                     │   │
│  │  - visual_files_var: Variable for images                │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PROMPT PREPARATION                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. Resolve {{variable}} references                     │   │
│  │  2. Load conversation history                           │   │
│  │  3. Add citation prompt if enabled                      │   │
│  │  4. Extract special tags (TASK_ANALYSIS, REFLECTION)    │   │
│  │  5. Handle visual files (images)                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM BUNDLE                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  - Multi-provider support (OpenAI, Anthropic, etc.)     │   │
│  │  - Automatic retry with backoff                         │   │
│  │  - Token counting & context fitting                     │   │
│  │  - Streaming support                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
                    Generated Content
                    (streaming or complete)
```

## LLMParam Class

```python
class LLMParam(ComponentParamBase):
    """
    LLM component parameters.
    """

    def __init__(self):
        super().__init__()
        # Model identification
        self.llm_id = ""  # Format: "model_name@provider"

        # Prompts
        self.sys_prompt = ""  # System prompt
        self.prompts = [{"role": "user", "content": "{sys.query}"}]

        # Generation parameters
        self.max_tokens = 0        # 0 = use model default
        self.temperature = 0       # 0-1, higher = more random
        self.top_p = 0             # Nucleus sampling
        self.presence_penalty = 0  # Penalize repetition
        self.frequency_penalty = 0 # Penalize frequent tokens

        # Output settings
        self.output_structure = None  # JSON schema for structured output
        self.cite = True              # Add citations to references

        # Visual input
        self.visual_files_var = None  # Variable containing images

    def check(self):
        """Validate parameters."""
        self.check_decimal_float(float(self.temperature), "[Agent] Temperature")
        self.check_decimal_float(float(self.presence_penalty), "[Agent] Presence penalty")
        self.check_decimal_float(float(self.frequency_penalty), "[Agent] Frequency penalty")
        self.check_nonnegative_number(int(self.max_tokens), "[Agent] Max tokens")
        self.check_decimal_float(float(self.top_p), "[Agent] Top P")
        self.check_empty(self.llm_id, "[Agent] LLM")
        self.check_empty(self.sys_prompt, "[Agent] System prompt")
        self.check_empty(self.prompts, "[Agent] User prompt")

    def gen_conf(self):
        """
        Generate LLM configuration dict.

        Only includes enabled parameters.
        """
        conf = {}
        if int(self.max_tokens) > 0 and getattr(self, "maxTokensEnabled", True):
            conf["max_tokens"] = int(self.max_tokens)
        if float(self.temperature) > 0 and getattr(self, "temperatureEnabled", True):
            conf["temperature"] = float(self.temperature)
        if float(self.top_p) > 0 and getattr(self, "topPEnabled", True):
            conf["top_p"] = float(self.top_p)
        if float(self.presence_penalty) > 0:
            conf["presence_penalty"] = float(self.presence_penalty)
        if float(self.frequency_penalty) > 0:
            conf["frequency_penalty"] = float(self.frequency_penalty)
        return conf
```

## LLM Class

```python
class LLM(ComponentBase):
    component_name = "LLM"

    def __init__(self, canvas, component_id, param: ComponentParamBase):
        super().__init__(canvas, component_id, param)

        # Initialize LLM Bundle
        self.chat_mdl = LLMBundle(
            self._canvas.get_tenant_id(),
            TenantLLMService.llm_id2llm_type(self._param.llm_id),
            self._param.llm_id,
            max_retries=self._param.max_retries,
            retry_interval=self._param.delay_after_error
        )
        self.imgs = []  # For visual files

    def get_input_elements(self) -> dict[str, Any]:
        """
        Extract {{variable}} references from prompts.
        """
        res = self.get_input_elements_from_text(self._param.sys_prompt)

        # Normalize prompts format
        if isinstance(self._param.prompts, str):
            self._param.prompts = [{"role": "user", "content": self._param.prompts}]

        # Extract from each prompt
        for prompt in self._param.prompts:
            d = self.get_input_elements_from_text(prompt["content"])
            res.update(d)
        return res
```

## Prompt Preparation Flow

```python
def _prepare_prompt_variables(self):
    """
    Prepare prompts with resolved variables.

    Process:
    1. Handle visual files if configured
    2. Resolve all variable references
    3. Build message history
    4. Add citation prompt if needed
    """
    # 1. Visual files handling
    if self._param.visual_files_var:
        self.imgs = self._canvas.get_variable_value(self._param.visual_files_var)
        self.imgs = [img for img in (self.imgs or [])
                    if img.startswith("data:image/")]

        # Switch to IMAGE2TEXT model if images present
        if self.imgs and TenantLLMService.llm_id2llm_type(self._param.llm_id) == LLMType.CHAT.value:
            self.chat_mdl = LLMBundle(
                self._canvas.get_tenant_id(),
                LLMType.IMAGE2TEXT.value,
                self._param.llm_id
            )

    # 2. Resolve variable values
    args = {}
    vars = self.get_input_elements()
    for k, o in vars.items():
        args[k] = o["value"]
        if not isinstance(args[k], str):
            args[k] = json.dumps(args[k], ensure_ascii=False)
        self.set_input_value(k, args[k])

    # 3. Build messages with history
    msg, sys_prompt = self._sys_prompt_and_msg(
        self._canvas.get_history(self._param.message_history_window_size)[:-1],
        args
    )

    # 4. Extract special prompt sections
    user_defined_prompt, sys_prompt = self._extract_prompts(sys_prompt)

    # 5. Add citation prompt if enabled
    if self._param.cite and self._canvas.get_reference()["chunks"]:
        sys_prompt += citation_prompt(user_defined_prompt)

    return sys_prompt, msg, user_defined_prompt
```

## Special Prompt Tags

```python
def _extract_prompts(self, sys_prompt):
    """
    Extract special XML-like tags from system prompt.

    Supported tags:
    - <TASK_ANALYSIS>: Task breakdown instructions
    - <PLAN_GENERATION>: Planning strategy
    - <REFLECTION>: Self-reflection guidelines
    - <CONTEXT_SUMMARY>: Context summarization
    - <CONTEXT_RANKING>: Ranking criteria
    - <CITATION_GUIDELINES>: Citation format

    Returns:
        (extracted_prompts: dict, cleaned_sys_prompt: str)
    """
    pts = {}
    for tag in ["TASK_ANALYSIS", "PLAN_GENERATION", "REFLECTION",
                "CONTEXT_SUMMARY", "CONTEXT_RANKING", "CITATION_GUIDELINES"]:
        r = re.search(rf"<{tag}>(.*?)</{tag}>", sys_prompt, flags=re.DOTALL|re.IGNORECASE)
        if not r:
            continue
        pts[tag.lower()] = r.group(1)
        sys_prompt = re.sub(rf"<{tag}>(.*?)</{tag}>", "", sys_prompt, flags=re.DOTALL|re.IGNORECASE)

    return pts, sys_prompt
```

## Invocation Modes

### 1. Structured Output Mode

```python
def _invoke_structured(self, prompt, msg):
    """
    Generate structured JSON output.

    Process:
    1. Add JSON schema to prompt
    2. Call LLM
    3. Parse and validate JSON
    4. Retry on parse errors
    """
    output_structure = self._param.outputs.get('structured')
    if not output_structure or not output_structure.get("properties"):
        return False

    schema = json.dumps(output_structure, ensure_ascii=False, indent=2)
    prompt += structured_output_prompt(schema)

    for _ in range(self._param.max_retries + 1):
        _, msg = message_fit_in(
            [{"role": "system", "content": prompt}, *msg],
            int(self.chat_mdl.max_length * 0.97)
        )

        ans = self._generate(msg)

        if ans.find("**ERROR**") >= 0:
            continue

        try:
            # Clean and parse JSON
            ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
            ans = re.sub(r"^.*```json", "", ans, flags=re.DOTALL)
            ans = re.sub(r"```\n*$", "", ans, flags=re.DOTALL)

            self.set_output("structured", json_repair.loads(ans))
            return True
        except Exception:
            msg.append({"role": "user", "content": "The answer can't be parsed as JSON"})

    return False
```

### 2. Streaming Output Mode

```python
def _stream_output(self, prompt, msg):
    """
    Stream LLM output token by token.

    Used when downstream is Message component
    for real-time user feedback.
    """
    _, msg = message_fit_in(
        [{"role": "system", "content": prompt}, *msg],
        int(self.chat_mdl.max_length * 0.97)
    )

    answer = ""
    for ans in self._generate_streamly(msg):
        if self.check_if_canceled("LLM streaming"):
            return

        if ans.find("**ERROR**") >= 0:
            if self.get_exception_default_value():
                self.set_output("content", self.get_exception_default_value())
                yield self.get_exception_default_value()
            else:
                self.set_output("_ERROR", ans)
            return

        yield ans
        answer += ans

    self.set_output("content", answer)
```

### 3. Batch Output Mode

```python
def _invoke_batch(self, prompt, msg):
    """
    Generate complete response before returning.

    Used when downstream needs full content
    or for exception handling.
    """
    for _ in range(self._param.max_retries + 1):
        if self.check_if_canceled("LLM processing"):
            return

        _, msg = message_fit_in(
            [{"role": "system", "content": prompt}, *msg],
            int(self.chat_mdl.max_length * 0.97)
        )

        ans = self._generate(msg)
        msg.pop(0)  # Remove system message for retry

        if ans.find("**ERROR**") >= 0:
            logging.error(f"LLM response error: {ans}")
            continue

        self.set_output("content", ans)
        return

    # All retries failed
    if self.get_exception_default_value():
        self.set_output("content", self.get_exception_default_value())
    else:
        self.set_output("_ERROR", error)
```

## Think Tag Handling

```python
def _generate_streamly(self, msg: list[dict], **kwargs):
    """
    Stream with <think>...</think> tag processing.

    Some models (like DeepSeek R1) use think tags
    for chain-of-thought reasoning.
    """
    ans = ""
    last_idx = 0
    endswith_think = False

    def delta(txt):
        nonlocal ans, last_idx, endswith_think
        delta_ans = txt[last_idx:]
        ans = txt

        # Handle think tag opening
        if delta_ans.find("<think>") == 0:
            last_idx += len("<think>")
            return "<think>"
        elif delta_ans.find("<think>") > 0:
            delta_ans = txt[last_idx:last_idx+delta_ans.find("<think>")]
            last_idx += delta_ans.find("<think>")
            return delta_ans

        # Handle think tag closing
        elif delta_ans.endswith("</think>"):
            endswith_think = True
        elif endswith_think:
            endswith_think = False
            return "</think>"

        last_idx = len(ans)
        if ans.endswith("</think>"):
            last_idx -= len("</think>")

        return re.sub(r"(<think>|</think>)", "", delta_ans)

    # Stream with image support
    if not self.imgs:
        for txt in self.chat_mdl.chat_streamly(
            msg[0]["content"], msg[1:], self._param.gen_conf(), **kwargs
        ):
            yield delta(txt)
    else:
        for txt in self.chat_mdl.chat_streamly(
            msg[0]["content"], msg[1:], self._param.gen_conf(),
            images=self.imgs, **kwargs
        ):
            yield delta(txt)
```

## Memory Management

```python
def add_memory(self, user: str, assist: str, func_name: str,
               params: dict, results: str, user_defined_prompt: dict = {}):
    """
    Add tool call summary to memory.

    Used by Agent component to maintain
    context across tool calls.
    """
    summ = tool_call_summary(
        self.chat_mdl, func_name, params, results, user_defined_prompt
    )
    logging.info(f"[MEMORY]: {summ}")
    self._canvas.add_memory(user, assist, summ)
```

## Output Structure

```python
# Standard outputs
outputs = {
    "content": "Generated text response",
    "_created_time": 1234567890.123,
    "_elapsed_time": 2.5,
    "_ERROR": None  # Error message if failed
}

# With structured output
outputs = {
    "content": "",
    "structured": {
        "field1": "value1",
        "field2": 123
    }
}

# With think tag (reasoning models)
outputs = {
    "content": "Final answer",
    "_think": "Chain of thought reasoning..."
}
```

## Usage in DSL

```json
{
    "LLM:Planning": {
        "obj": {
            "component_name": "LLM",
            "params": {
                "llm_id": "gpt-4o@OpenAI",
                "sys_prompt": "You are a helpful assistant.",
                "prompts": [
                    {"role": "user", "content": "Question: {{sys.query}}"}
                ],
                "temperature": 0.7,
                "max_tokens": 2048,
                "cite": true,
                "output_structure": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "confidence": {"type": "number"}
                    }
                }
            }
        },
        "downstream": ["Message:Output"],
        "upstream": ["begin"]
    }
}
```

## Streaming Decision Logic

```python
def _invoke(self, **kwargs):
    """
    Main invocation with mode selection.
    """
    prompt, msg, _ = self._prepare_prompt_variables()

    # Mode 1: Structured output
    if self._param.output_structure:
        return self._invoke_structured(prompt, msg)

    # Mode 2: Stream to Message component
    downstreams = self._canvas.get_component(self._id)["downstream"]
    ex = self.exception_handler()
    if any([self._canvas.get_component_obj(cid).component_name.lower() == "message"
            for cid in downstreams]) and not (ex and ex["goto"]):
        # Return partial function for lazy streaming
        self.set_output("content", partial(self._stream_output, prompt, msg))
        return

    # Mode 3: Batch output
    return self._invoke_batch(prompt, msg)
```

## Related Files

- `/agent/component/llm.py` - LLM component
- `/agent/component/base.py` - ComponentBase class
- `/api/db/services/llm_service.py` - LLMBundle
- `/rag/prompts/generator.py` - Prompt utilities
