# llm_client.py
from __future__ import annotations
import os, json, time, typing as T
import requests, pathlib


# Optional: strict validation if you pass a JSON Schema
try:
    import jsonschema  # pip install jsonschema
    _HAS_JSONSCHEMA = True
except Exception:
    _HAS_JSONSCHEMA = False

# ---------- Config via env ----------
from dotenv import load_dotenv
load_dotenv()
PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()  # "anthropic" | "ollama"
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

DEFAULT_TIMEOUT = float(os.getenv("LLM_TIMEOUT_S", "600"))
MAX_RETRIES = int(os.getenv("LLM_RETRIES", "2"))

# ---------- Public API ----------
def llm_call(
    prompt: str,
    *,
    json_schema: dict | None = None,
    tool_spec: dict | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1200,
    provider: str | None = None,
) -> T.Union[str, dict]:
    """
    Unified LLM call.
    - If json_schema is provided: returns a dict (validated if jsonschema is installed).
    - Else: returns plain text string.
    """
    prov = (provider or PROVIDER).lower()
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            if prov == "anthropic":
                return _anthropic_call(prompt, tool_spec, temperature, max_tokens)
            elif prov == "ollama":
                return _ollama_call(prompt, json_schema, temperature, max_tokens)
            else:
                raise ValueError(f"Unknown provider: {prov}")
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(0.75 * (attempt + 1))
            else:
                raise

    # unreachable, but keeps type-checkers happy
    raise RuntimeError(f"LLM call failed: {last_err}")

# ---------- Anthropic ----------
def _anthropic_call(prompt: str, tool_spec: dict | None,
                    temperature: float, max_tokens: int):
    """
    tools mode: tool_spec must be a SINGLE tool dict:
      {"name": "...", "description": "...", "input_schema": {...}}
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    from anthropic import Anthropic
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    messages_kwargs = dict(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    if tool_spec:
        # Claude expects: tools=[{name, description, input_schema}]
        messages_kwargs["tools"] = [tool_spec]
        # Optional but helpful: force tool choice
        messages_kwargs["tool_choice"] = {"type": "tool", "name": tool_spec["name"]}

    resp = client.messages.create(**messages_kwargs)

    if not tool_spec:
        # No tools → return plain text
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()
        return text

    # Tools mode: find the tool_use block
    for block in resp.content:
        if getattr(block, "type", "") == "tool_use" and block.name == tool_spec["name"]:
            data = block.input  # <-- already a dict per input_schema
            # Validate against the *input_schema* (not the whole tool spec)
            if _HAS_JSONSCHEMA:
                jsonschema.validate(data, tool_spec["input_schema"])
            return data

    raise RuntimeError("No tool_use block returned by Claude")

# ---------- Ollama ----------
def _ollama_call(
    prompt: str,
    system: str | None,
    json_schema: dict | None,
    temperature: float,
    max_tokens: int,
) -> T.Union[str, dict]:
    """
    Uses Ollama's /api/generate (non-stream) for simplicity.
    - Ollama cannot *enforce* a JSON Schema; we prompt it to comply, then parse+validate.
    """
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    if json_schema:
        instruction = (
            "Return ONLY valid JSON that strictly matches this JSON Schema. "
            "Do not include backticks or extra text.\n\nSCHEMA:\n"
            f"{json.dumps(json_schema, indent=2)}\n\nPROMPT:\n{prompt}"
        )
        req = {
            "model": OLLAMA_MODEL,
            "prompt": _build_system_user(system, instruction),
            "options": {"temperature": temperature},
            "format": "json",      # asks model to output JSON
            "stream": False,
        }
    else:
        req = {
            "model": OLLAMA_MODEL,
            "prompt": _build_system_user(system, prompt),
            "options": {"temperature": temperature},
            "stream": False,
        }

    r = requests.post(url, json=req, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    out = r.json()  # {'model':..., 'response': '...', ...}
    text = out.get("response", "")

    if not json_schema:
        return text.strip()

    data = _best_effort_json_parse(text)
    if _HAS_JSONSCHEMA:
        jsonschema.validate(data, json_schema)
    return data

# ---------- Utils ----------
def _build_system_user(system: str | None, user: str) -> str:
    if system:
        return f"System:\n{system}\n\nUser:\n{user}"
    return user

def _best_effort_json_parse(s: str, dump_dir: str = "./mcp_data/bad_json") -> dict:
    s = s.strip()
    # Try straight parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try to find the first {...} block
    import re
    m = re.search(r"\{(?:.|\n)*\}", s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    dump_path = pathlib.Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    fname = dump_path / f"llm_raw_{ts}.txt"
    fname.write_text(s, encoding="utf-8")

    # 4. Return a sentinel dict so caller knows it failed
    return {"_error": "Failed to parse JSON", "_raw_file": str(fname)}
    
