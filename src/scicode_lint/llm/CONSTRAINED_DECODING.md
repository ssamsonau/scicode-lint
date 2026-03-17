# Constrained Decoding: Why `guided_json`

Design decision: why we use vLLM's `guided_json` instead of OpenAI-standard `response_format: json_schema`.

Both achieve schema-constrained JSON output, but they behave differently with thinking models.

### The two approaches

**`guided_json`** (vLLM-specific, what we use):
```python
extra_body={"guided_json": schema}
```

**`response_format: json_schema`** (OpenAI API standard):
```python
response_format={"type": "json_schema", "json_schema": {...}}
```

### How constrained decoding works

Both use the same backend (Outlines/XGrammar in vLLM):

1. Parse schema → build FSM (finite state machine)
2. Track valid next tokens at each step
3. Mask invalid tokens before sampling

```
Schema: {"detected": bool}

After '{"detected": ' → only allow: true, false
After '{"detected": true' → only allow: , or }
```

### The difference: prefix handling

**`guided_json`** - allows any prefix before JSON:
```
<think>                    ← unconstrained
Let me analyze line 42...
</think>                   ← unconstrained
{                          ← constraints start here
"detected": true
}
```

**`json_schema`** - requires immediate JSON:
```
{                          ← must start immediately
"detected": true           ← no thinking possible
}
```

### Why this matters for Qwen3

Qwen3 outputs visible `<think>...</think>` blocks before answering. With `json_schema`, these get suppressed because `<` is not valid JSON.

| Approach | Thinking preserved | Accuracy |
|----------|-------------------|----------|
| `guided_json` | Yes | ~99% |
| `json_schema` | No | ~78% |

~20% accuracy drop without thinking phase.

### OpenAI models work differently

OpenAI's o1/o3 have **hidden** reasoning tokens (internal, not in output). So `json_schema` works fine - reasoning happens before output begins.

Qwen3's reasoning is **visible** → suppressed by immediate JSON constraint.

### Our choice

We use `guided_json` because:
1. Preserves thinking phase (critical for accuracy)
2. Same underlying constrained decoding quality
3. We're committed to vLLM (no portability concern)

### Other vLLM constrained decoding options

| Option | Constrains to | Allows thinking prefix | Use case |
|--------|---------------|----------------------|----------|
| `guided_json` | JSON schema | With reasoning parser¹ | Structured output |
| `guided_choice` | One of N strings | No | Simple classification |
| `guided_regex` | Regex pattern | With reasoning parser¹ | IDs, formatted strings |
| `guided_grammar` | Context-free grammar | Unknown | SQL, custom formats |

¹ Requires v0 engine with `--enable-reasoning --reasoning-parser`. See [vLLM reasoning docs](https://docs.vllm.ai/en/v0.8.4/features/reasoning_outputs.html).

**Important**: By default, all guided decoding options apply constraints from the first token. The `guided_json` vs `json_schema` difference we observe with Qwen3 may be implementation-specific. Both `guided_choice` and `guided_regex` require immediate constrained output without the reasoning parser, skipping the thinking phase.

**Deprecation note**: Starting with vLLM ~0.8.5+, `guided_*` parameters are being deprecated in favor of unified `structured_outputs` format.

### Pydantic schemas are mandatory

Always use Pydantic models to generate JSON schemas for `guided_json`. Hand-written schemas may have subtle issues that cause unreliable constraint enforcement.

```python
from pydantic import BaseModel, Field

class FilterResult(BaseModel):
    is_pipeline: bool = Field(description="True if code is ML pipeline")

# Use Pydantic-generated schema
extra_body={"guided_json": FilterResult.model_json_schema()}
```

Pydantic schemas include:
- Proper `title` fields for each property
- Consistent `type` annotations
- `required` array automatically generated
- Field descriptions for better model understanding

All use same Outlines/XGrammar backend.

### References

- vLLM structured outputs: https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html
- vLLM reasoning outputs: https://docs.vllm.ai/en/v0.8.4/features/reasoning_outputs.html
- Outlines: https://github.com/dottxt-ai/outlines
- XGrammar: https://github.com/mlc-ai/xgrammar
