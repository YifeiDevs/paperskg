# llm_client.py
from __future__ import annotations
import os, json, time, typing as T, logging
import requests, pathlib

# Set up logger for this module
logger = logging.getLogger(__name__)

# Optional: strict validation if you pass a JSON Schema
try:
    import jsonschema  # pip install jsonschema
    _HAS_JSONSCHEMA = True
    logger.info("JSON Schema validation available")
except Exception:
    _HAS_JSONSCHEMA = False
    logger.warning("JSON Schema validation not available")

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

logger.info(f"LLM client initialized - Provider: {PROVIDER}, Model: {ANTHROPIC_MODEL if PROVIDER == 'anthropic' else OLLAMA_MODEL}")

# ---------- Public API ----------
def llm_call(
    prompt: str,
    *,
    json_schema: dict | None = None,
    tool_spec: dict | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024*1024,
    provider: str | None = None,
) -> T.Union[str, dict]:
    """
    Unified LLM call.
    - If json_schema is provided: returns a dict (validated if jsonschema is installed).
    - Else: returns plain text string.
    """

    prov = (provider or PROVIDER).lower()
    logger.info(f"LLM call initiated - Provider: {prov}, Temperature: {temperature}, Max tokens: {max_tokens}")
    # logger.info(f"{prompt = }")
    logger.info(f"{tool_spec = }")

    if tool_spec:
        logger.debug(f"Using tool spec: {tool_spec.get('name', 'unknown')}")
    elif json_schema:
        logger.debug("Using JSON schema mode")
    else:
        logger.debug("Using plain text mode")
    
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.debug(f"LLM call attempt {attempt + 1}/{MAX_RETRIES + 1}")
            
            if prov == "anthropic":
                logger.info("Anthropic Call Start")
                result = _anthropic_call(prompt, tool_spec, temperature, max_tokens)
                logger.info(f"{result = }")
                logger.info(f"LLM call successful on attempt {attempt + 1}")
                return result
            elif prov == "ollama":
                result = _ollama_call(prompt, json_schema, temperature, max_tokens)
                logger.info(f"LLM call successful on attempt {attempt + 1}")
                return result
            else:
                raise ValueError(f"Unknown provider: {prov}")
                
        except Exception as e:
            last_err = e
            logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
            
            if attempt < MAX_RETRIES:
                sleep_time = 0.75 * (attempt + 1)
                logger.debug(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"All LLM call attempts failed. Last error: {e}")
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
    logger.debug(f"Anthropic call started - Model: {ANTHROPIC_MODEL}")
    
    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set")
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.debug("Anthropic client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic client: {e}")
        raise

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
        logger.debug(f"Using tool mode with tool: {tool_spec['name']}")
    else:
        logger.debug("Using text mode")

    logger.debug(f"Sending request to Anthropic - Prompt length: {len(prompt)} characters")
    
    try:
        resp = client.messages.create(**messages_kwargs)
        logger.info("Anthropic API call successful")
    except Exception as e:
        logger.error(f"Anthropic API call failed: {e}")
        raise

    if not tool_spec:
        # No tools → return plain text
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()
        logger.debug(f"Returning text response, length: {len(text)} characters")
        return text

    # Tools mode: find the tool_use block
    logger.debug("Processing tool response")
    for block in resp.content:
        if getattr(block, "type", "") == "tool_use" and block.name == tool_spec["name"]:
            # 调试：打印原始输入
            logger.debug(f"Raw block.input type = {type(block.input)}")
            logger.debug(f"Raw block.input = {str(block.input)[:500]}...")
            
            # 处理字符串情况
            if isinstance(block.input, str):
                logger.debug(f"Parsing JSON string...")
                cleaned = block.input.strip().rstrip(',')
                logger.debug(f"Cleaned input = {cleaned[:200]}...{cleaned[-50:]}")
                data = json.loads(cleaned)
                logger.debug(f"Parsed data type = {type(data)}")
            else:
                data = block.input
                logger.debug(f"Input already parsed")
            
            # 断言：确保是字典
            assert isinstance(data, dict), f"Expected dict, got {type(data)}"
            logger.debug(f"Final data keys = {list(data.keys())}")
            
            # Schema 验证
            if _HAS_JSONSCHEMA:
                jsonschema.validate(data, tool_spec["input_schema"])
                logger.debug("Schema validation passed")
            
            return data

    logger.error("No tool_use block returned by Claude")
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
    logger.debug(f"Ollama call started - Model: {OLLAMA_MODEL}, URL: {OLLAMA_URL}")
    
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    
    if json_schema:
        logger.debug("Using JSON schema mode for Ollama")
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
        logger.debug("Using text mode for Ollama")
        req = {
            "model": OLLAMA_MODEL,
            "prompt": _build_system_user(system, prompt),
            "options": {"temperature": temperature},
            "stream": False,
        }

    logger.debug(f"Sending request to Ollama - Prompt length: {len(prompt)} characters")
    
    try:
        r = requests.post(url, json=req, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        logger.info("Ollama API call successful")
    except Exception as e:
        logger.error(f"Ollama API call failed: {e}")
        raise

    out = r.json()  # {'model':..., 'response': '...', ...}
    text = out.get("response", "")
    logger.debug(f"Ollama response length: {len(text)} characters")

    if not json_schema:
        return text.strip()

    logger.debug("Parsing JSON response from Ollama")
    data = _best_effort_json_parse(text)
    
    if _HAS_JSONSCHEMA:
        try:
            jsonschema.validate(data, json_schema)
            logger.debug("Ollama response validated against schema")
        except Exception as e:
            logger.warning(f"Ollama response failed schema validation: {e}")
    else:
        logger.debug("Schema validation skipped (jsonschema not available)")
    
    return data

# ---------- Utils ----------
def _build_system_user(system: str | None, user: str) -> str:
    logger.debug("Building system/user prompt")
    if system:
        return f"System:\n{system}\n\nUser:\n{user}"
    return user

def _best_effort_json_parse(s: str, dump_dir: str = "./mcp_data/bad_json") -> dict:
    logger.debug(f"Attempting to parse JSON from text, length: {len(s)} characters")
    s = s.strip()
    
    # Try straight parse
    try:
        result = json.loads(s)
        logger.debug("JSON parsed successfully on first attempt")
        return result
    except Exception as e:
        logger.debug(f"Direct JSON parse failed: {e}")

    # Try to find the first {...} block
    import re
    m = re.search(r"\{(?:.|\n)*\}", s)
    if m:
        try:
            result = json.loads(m.group(0))
            logger.debug("JSON parsed successfully from extracted block")
            return result
        except Exception as e:
            logger.debug(f"Block JSON parse failed: {e}")

    # Save raw response for debugging
    logger.warning("Failed to parse JSON, saving raw response for debugging")
    dump_path = pathlib.Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    fname = dump_path / f"llm_raw_{ts}.txt"
    
    try:
        fname.write_text(s, encoding="utf-8")
        logger.info(f"Raw LLM response saved to: {fname}")
    except Exception as e:
        logger.error(f"Failed to save raw response: {e}")

    # Return a sentinel dict so caller knows it failed
    return {"_error": "Failed to parse JSON", "_raw_file": str(fname)}
    
