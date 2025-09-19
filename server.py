from mcp.server.fastmcp import FastMCP, Context
from schemas import *
from typing import List, Dict, Any, Optional
import pathlib, json, re, logging

# Set up logger for this module
logger = logging.getLogger(__name__)

# Optional: use your unified client if present; else fall back to a stub
try:
    from llm_client import llm_call
    logger.info("LLM client imported successfully")
except Exception as e:
    llm_call = None  # we'll stub if missing
    logger.warning(f"LLM client not available, using stub mode: {e}")

mcp = FastMCP(name="PapersKG")
DATA = pathlib.Path("./mcp_data")
DATA.mkdir(exist_ok=True)
logger.info(f"Data directory initialized: {DATA}")

# in-memory index of resource URIs -> file path
papers_index: Dict[str, pathlib.Path] = {}
logger.info("Papers index initialized")

# ---------- helpers ----------
def pdf_to_text(pdf_path: pathlib.Path) -> str:
    logger.info(f"Starting PDF text extraction from: {pdf_path}")
    
    # Try PyMuPDF first
    try:
        import fitz  # PyMuPDF
        logger.debug("Using PyMuPDF for PDF extraction")
        doc = fitz.open(pdf_path)
        chunks = []
        page_count = len(doc)
        logger.info(f"Processing {page_count} pages with PyMuPDF")
        
        for i, p in enumerate(doc):
            chunks.append(p.get_text("text"))
            if (i + 1) % 10 == 0:  # Log every 10 pages
                logger.debug(f"Processed {i + 1}/{page_count} pages")
        
        text = "\n".join(chunks)
        logger.info(f"PyMuPDF extraction completed, text length: {len(text)} characters")
    except Exception as e:
        logger.warning(f"PyMuPDF failed, falling back to pdfminer: {e}")
        # Fallback: pdfminer.six
        from pdfminer.high_level import extract_text
        text = extract_text(str(pdf_path))
        logger.info(f"Pdfminer extraction completed, text length: {len(text)} characters")
    
    # light cleanup
    original_length = len(text)
    text = re.sub(r"-\n(\w)", r"\1", text)  # de-hyphenate
    text = re.sub(r"[ \t]+\n", "\n", text)
    logger.debug(f"Text cleanup completed, length changed: {original_length} -> {len(text)}")
    
    return text

def _cache_path(resource_uri: str, suffix: str) -> pathlib.Path:
    cache_path = DATA / f"{resource_uri.replace(':','_')}{suffix}"
    logger.debug(f"Generated cache path for {resource_uri}: {cache_path}")
    return cache_path

def _assert_rid(resource_uri: str) -> pathlib.Path:
    logger.debug(f"Validating resource URI: {resource_uri}")
    if resource_uri not in papers_index:
        logger.error(f"Unknown resource_uri: {resource_uri}")
        raise ValueError(f"Unknown resource_uri: {resource_uri}")
    
    path = papers_index[resource_uri]
    logger.debug(f"Resource URI validated, path: {path}")
    return path

# ---------- resources ----------
@mcp.resource("paper://{rid}")
def read_paper(rid: str) -> str:
    """
    Read the paper's cached/extracted text by rid.
    (Keep this read-only; do heavy work in tools.)
    """
    logger.info(f"MCP resource request: reading paper {rid}")
    
    path = papers_index.get(rid)
    if not path:
        logger.error(f"Paper resource not found: {rid}")
        raise ValueError(f"Unknown rid: {rid}")
    # TODO: replace with your cached text; for now, show file path as proof
    return f"PDF path: {path}"
    # e.g., return (DATA / f"{rid}.txt").read_text()

# ---------- tools ----------
@mcp.tool()
def index_folder(path: str) -> List[str]:
    """
    Register all PDFs in a folder as MCP resources. Returns resource URIs.
    """
    logger.info(f"MCP tool called: index_folder with path: {path}")
    
    p = pathlib.Path(path).expanduser().resolve()
    logger.debug(f"Resolved path: {p}")
    
    if not p.exists():
        logger.error(f"Path not found: {p}")
        raise ValueError(f"Path not found: {p}")
    
    logger.info(f"Scanning for PDF files in: {p}")
    pdf_files = list(p.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    out: List[str] = []
    for pdf in pdf_files:
        rid = f"paper:{pdf.stem}"
        original_rid = rid
        
        # avoid collisions
        i = 1
        while rid in papers_index:
            rid = f"paper:{pdf.stem}-{i}"; i += 1
        if rid != original_rid:
            logger.debug(f"Collision resolved: {original_rid} -> {rid}")
        
        papers_index[rid] = pdf.resolve()
        out.append(rid)
        logger.debug(f"Indexed PDF: {pdf.name} as {rid}")
    
    logger.info(f"Successfully indexed {len(out)} papers: {out}")
    return out

@mcp.tool()
def ingest_paper(resource_uri: str) -> str:
    """
    Extract and cache clean text for a paper. Returns cache path.
    """
    logger.info(f"MCP tool called: ingest_paper for resource: {resource_uri}")
    
    path = _assert_rid(resource_uri)
    logger.info(f"Starting text extraction for: {path}")
    
    text = pdf_to_text(path)
    out = _cache_path(resource_uri, ".txt")
    
    try:
        out.write_text(text, encoding="utf-8")
        logger.info(f"Text cached successfully at: {out}")
        logger.info(f"Cached text length: {len(text)} characters")
    except Exception as e:
        logger.error(f"Failed to cache text: {e}")
        raise
    
    return str(out)

@mcp.tool()
def extract_entities(resource_uri: str) -> List[Material]:
    """
    Extract materials/entities from the cached text via an LLM tool call.
    Expects llm_call(...) to support Anthropic 'tools' mode where we pass
    a single tool spec and receive tool_use.input (a dict) back.
    """
    logger.info(f"MCP tool called: extract_entities for resource: {resource_uri}")
    
    _assert_rid(resource_uri)

    # Load cached or on-the-fly extracted text
    txt_path = _cache_path(resource_uri, ".txt")
    # text = txt_path.read_text() if txt_path.exists() else pdf_to_text(papers_index[resource_uri])
    text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else pdf_to_text(papers_index[resource_uri])
    logger.debug(f"Loaded cached text, length: {len(text)} characters")
    # If no LLM hook, return a tiny stub so the pipeline runs
    if not llm_call:
        logger.warning("No LLM client available, using stub data")
        mats = [{"id": "mat:ysz", "name": "Yttria-stabilized zirconia", "formula": "ZrO2:Y2O3"}]
        ents_path = _cache_path(resource_uri, "_entities.json")
        ents_path.write_text(json.dumps({"materials": mats}, indent=2))
        logger.info("Stub entities saved, returning Material objects")
        return [Material(**m) for m in mats]

    # ----- Define ONE tool spec (not a list). Anthropic will receive tools=[tool_spec].
    MATERIAL_TOOL = {
        "name": "material_extraction_from_paper",
        "description": (
            "Retrieve information about materials discussed in the paper. "
            "Only include materials that were experimentally synthesized and measured."
        ),
        "input_schema" : {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "am.material.v1",
  "type": "object",
  "title": "am material extraction",
  "description": "Structured facts about materials processed by LPBF in a scientific paper.",
  "properties": {
    "materials": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "name"],
        "additionalProperties": True,
        "properties": {
          "id": {"type": "string", "description": "Stable identifier, e.g., mat:17-4PH"},
          "name": {"type": "string", "description": "Human-readable material name (e.g., 316L, Ti-6Al-4V, Inconel 718)"},
          "material_system": {"type": ["string","null"], "description": "Fe-Cr-Ni, Ti-Al-V, Ni-Cr-Mo, etc."},

          "composition": {
            "type": ["array","null"],
            "description": "Elemental composition if reported",
            "items": {
              "type": "object",
              "required": ["element", "amount", "unit"],
              "additionalProperties": False,
              "properties": {
                "element": {"type": "string", "description": "e.g., Fe, Cr, Ni"},
                "amount": {"type": "number"},
                "unit": {"type": "string", "enum": ["wt%", "at%", "ppm"]}
              }
            }
          },

          "powder": {
            "type": ["object","null"],
            "additionalProperties": True,
            "properties": {
              "supplier": {"type": ["string","null"]},
              "lot": {"type": ["string","null"]},
              "atomization": {"type": ["string","null"], "description": "e.g., gas, plasma"},
              "size_d10_um": {"type": ["number","null"]},
              "size_d50_um": {"type": ["number","null"]},
              "size_d90_um": {"type": ["number","null"]},
              "morphology": {"type": ["string","null"]},
              "purity_pct": {"type": ["number","null"]},
              "oxygen_ppm": {"type": ["number","null"]},
              "nitrogen_ppm": {"type": ["number","null"]}
            }
          },

          "lpbf_process": {
            "type": ["object","null"],
            "additionalProperties": True,
            "properties": {
              "machine": {"type": ["string","null"], "description": "e.g., EOS M290"},
              "laser_type": {"type": ["string","null"], "description": "e.g., fiber, Yb:YAG"},
              "laser_power_W": {"type": ["number","null"]},
              "scan_speed_mm_s": {"type": ["number","null"]},
              "hatch_spacing_um": {"type": ["number","null"]},
              "layer_thickness_um": {"type": ["number","null"]},
              "preheat_temp_C": {"type": ["number","null"]},
              "scan_strategy": {"type": ["string","null"], "description": "e.g., stripe, chessboard, rotation 67°"},
              "atmosphere": {"type": ["string","null"], "description": "e.g., Ar, N2"},
              "oxygen_ppm": {"type": ["number","null"]},
              "build_plate": {"type": ["string","null"], "description": "material or temperature"},
              "recoater": {"type": ["string","null"]},
              "relative_density_pct": {"type": ["number","null"]},
              "energy_density_J_mm3": {"type": ["number","null"], "description": "volumetric energy density if reported"},
              "part_orientation": {"type": ["string","null"], "description": "e.g., 0°, 90° to build direction"},
              "geometry": {"type": ["string","null"], "description": "coupon geometry if relevant"}
            }
          },

          "post_processing": {
            "type": ["array","null"],
            "items": {
              "type": "object",
              "additionalProperties": True,
              "properties": {
                "type": {"type": ["string","null"], "description": "HIP, stress-relief, solution, aging, anneal"},
                "temp_C": {"type": ["number","null"]},
                "time_h": {"type": ["number","null"]},
                "pressure_MPa": {"type": ["number","null"]},
                "atmosphere": {"type": ["string","null"]},
                "cooling": {"type": ["string","null"], "description": "e.g., air, furnace, water quench"},
                "notes": {"type": ["string","null"]}
              }
            }
          },

          "experimental_methods": {
            "type": ["array","null"],
            "description": "Characterization & test methods actually used",
            "items": {
              "type": "object",
              "required": ["name"],
              "additionalProperties": True,
              "properties": {
                "name": {
                  "type": "string",
                  "description": "e.g., tensile, hardness, fatigue, SEM, EBSD, XRD, CT, density (Archimedes)"
                },
                "standard": {"type": ["string","null"], "description": "e.g., ASTM E8/E8M"},
                "instrument": {"type": ["string","null"], "description": "model/manufacturer if given"},
                "params": {"type": ["object","null"], "additionalProperties": True,
                  "description": "method-specific parameters (e.g., strain_rate_s, load, step size)"}
              }
            }
          },

          "properties": {
            "type": ["array","null"],
            "description": "Measured properties with conditions and provenance",
            "items": {
              "type": "object",
              "required": ["name","value","unit"],
              "additionalProperties": True,
              "properties": {
                "name": {
                  "type": "string",
                  "description": "e.g., yield_strength, ultimate_tensile_strength, elongation, hardness_HV, density, porosity_pct, fatigue_limit"
                },
                "value": {"type": "number"},
                "unit": {"type": "string"},
                "conditions": {
                  "type": ["object","null"],
                  "additionalProperties": True,
                  "properties": {
                    "temperature_C": {"type": ["number","null"]},
                    "strain_rate_s": {"type": ["number","null"]},
                    "direction": {"type": ["string","null"], "description": "relative to build (e.g., vertical, horizontal)"},
                    "specimen_orientation": {"type": ["string","null"]},
                    "relative_density_pct": {"type": ["number","null"]},
                    "porosity_pct": {"type": ["number","null"]},
                    "surface_condition": {"type": ["string","null"], "description": "as-built, machined, polished"},
                    "cycle_R_ratio": {"type": ["number","null"], "description": "fatigue R"},
                    "frequency_Hz": {"type": ["number","null"]}
                  }
                },
                "method_ref": {
                  "type": ["string","null"],
                  "description": "name of the experimental method entry that produced this property"
                },
                "provenance": {
                  "type": ["object","null"],
                  "additionalProperties": True,
                  "properties": {
                    "page_span": {"type": ["string","null"]},
                    "figure": {"type": ["string","null"]},
                    "table": {"type": ["string","null"]},
                    "quote": {"type": ["string","null"]}
                  }
                }
              }
            }
          }
        }
      }
    },

    "paper": {
      "type": ["object","null"],
      "additionalProperties": True,
      "description": "Optional metadata for the source paper",
      "properties": {
        "title": {"type": ["string","null"]},
        "doi": {"type": ["string","null"]},
        "year": {"type": ["integer","null"]}
      }
    }
  },
  "required": ["materials"],
  "additionalProperties": False
}

    }

    logger.info("Preparing LLM extraction query")
    
    # Prompt: clearly ask the model to use the tool and return only fields that match the schema.
    query = f"""
You are an expert scientific information extractor.
Use the tool 'material_extraction_from_paper' to return a JSON object that matches its input_schema.
If a field is unknown, omit it rather than guessing.

<document>
{text[:]}
</document>"""

    logger.debug(f"Query prepared, text length: {len(text)} characters")

    # ----- Call the LLM in tools mode
    # Expectation: llm_call will pass tools=[MATERIAL_TOOL] to Anthropic, force the tool, and
    # return the tool_use.input dict. (See the helper we discussed.)
    logger.info("Calling LLM for entity extraction")
    
    try:
        out = llm_call(
            prompt=query,
            tool_spec=MATERIAL_TOOL,
            temperature=0.0,
            max_tokens=1200,
        )
        logger.info("LLM call successful with tool_spec")
    except TypeError as e:
        logger.warning(f"LLM call with tool_spec failed, trying json_schema fallback: {e}")
        # Fallback for an older llm_call signature: if it only accepts json_schema,
        # we still pass the single tool spec; your llm_call should wrap it into tools=[...]
        # and return the tool_use.input dict.
        out = llm_call(
            prompt=query,
            json_schema=MATERIAL_TOOL,
            temperature=0.0,
            max_tokens=1200,
        )
        logger.info("LLM call successful with json_schema fallback")

    # Defensive handling: if your llm_call returns a sentinel instead of a dict, bail to stub
    if not isinstance(out, dict):
        logger.warning(f"LLM returned non-dict result: {type(out)}, using empty materials")
        out = {"materials": []}
    # mats = out.get("materials", []) or []
    extraction = LPBFExtraction(**out)

    # Persist and return as Pydantic models
    ents_path = _cache_path(resource_uri, "_entities.json")
    # ents_path.write_text(extraction.model_dump_json(indent=2))
    ents_path.write_text(extraction.model_dump_json(indent=2), encoding="utf-8")
    return extraction.materials


@mcp.tool()
def extract_relations(resource_uri: str) -> Dict[str, Any]:
    """
    Produce simple relations (stub). Replace with LLM-based relation extraction.
    """
    _assert_rid(resource_uri)
    ents_path = _cache_path(resource_uri, "_entities.json")
    if not ents_path.exists():
        raise RuntimeError("Run extract_entities first.")
    # ents = json.loads(ents_path.read_text())
    ents = json.loads(ents_path.read_text(encoding="utf-8"))
    mats = ents.get("materials", [])
    # stub: no real properties/process; just return empty or demo edge
    rels = {"relations": []}
    if mats:
        rels["relations"].append({
            "src": mats[0]["id"], "rel": "MENTIONED_IN", "tgt": resource_uri, "attrs": {}
        })
    rel_path = _cache_path(resource_uri, "_relations.json")
    # rel_path.write_text(json.dumps(rels, indent=2))
    rel_path.write_text(json.dumps(rels, indent=2), encoding="utf-8")
    return rels

@mcp.tool()
def export_graph(format: str = "neo4j") -> Dict[str, str]:
    """
    Export all cached LPBF extractions to Neo4j-importable CSVs:
      - Nodes: Paper, Material, Process, Method, Property
      - Edges: Paper MENTIONS Material/Process/Method/Property
               Material PROCESSED_BY Process
               Material HAS_PROPERTY Property  (attrs hold value/unit/conditions/provenance)
               Method REPORTS Property (optional; if method_ref is present)
    """
    logger.info(f"MCP tool called: export_graph with format: {format}")
    
    import csv, json
    nodes, edges = {}, []   # nodes keyed by id
    
    entity_files = list(DATA.glob("*_entities.json"))
    logger.info(f"Found {len(entity_files)} entity files to process")
    
    for f in entity_files:
        logger.debug(f"Processing entity file: {f}")
        base = f.stem.replace("_entities","")
        paper_id = base.replace("_",":",1)  # "paper:xxxx"
        # extraction = LPBFExtraction(**json.loads(f.read_text()))
        extraction = LPBFExtraction(**json.loads(f.read_text(encoding="utf-8")))
        logger.debug(f"Loaded extraction for paper: {paper_id}")

        # paper node
        nodes.setdefault(paper_id, {"id": paper_id, "type": "Paper", "props": extraction.paper.model_dump() if extraction.paper else {}})
        logger.debug(f"Added paper node: {paper_id}")

        materials_count = len(extraction.materials)
        logger.debug(f"Processing {materials_count} materials for paper {paper_id}")

        for mat in extraction.materials:
            nodes.setdefault(mat.id, {"id": mat.id, "type": "Material", "props": {"name": mat.name, "formula": mat.formula, "material_system": mat.material_system}})
            edges.append({"src": paper_id, "rel": "MENTIONS", "tgt": mat.id, "attrs": {}})

            # process (one per material if present)
            if mat.lpbf_process:
                proc_id = f"proc:{mat.id}"
                nodes.setdefault(proc_id, {"id": proc_id, "type": "LPBFProcess", "props": mat.lpbf_process.model_dump(exclude_none=True)})
                edges.append({"src": mat.id, "rel": "PROCESSED_BY", "tgt": proc_id, "attrs": {}})
                edges.append({"src": paper_id, "rel": "MENTIONS", "tgt": proc_id, "attrs": {}})
                logger.debug(f"Added LPBF process node: {proc_id}")

            # methods
            meth_name_to_id = {}
            if mat.experimental_methods:
                logger.debug(f"Processing {len(mat.experimental_methods)} experimental methods for material {mat.id}")
                for m in mat.experimental_methods:
                    mid = f"method:{mat.id}:{(m.name or 'unknown').lower()}"
                    meth_name_to_id[m.name] = mid
                    nodes.setdefault(mid, {"id": mid, "type": "Method", "props": m.model_dump(exclude_none=True)})
                    edges.append({"src": paper_id, "rel": "MENTIONS", "tgt": mid, "attrs": {}})

            # properties
            if mat.properties:
                logger.debug(f"Processing {len(mat.properties)} properties for material {mat.id}")
                for p in mat.properties:
                    pid = f"prop:{mat.id}:{(p.name).lower()}"
                    if pid not in nodes:
                        nodes[pid] = {"id": pid, "type": "Property", "props": {"name": p.name}}
                    # main fact on edge attrs
                    attrs = {
                        "value": p.value, "unit": p.unit,
                        **(p.conditions.model_dump(exclude_none=True) if p.conditions else {}),
                        **(p.provenance or {})
                    }
                    edges.append({"src": mat.id, "rel": "HAS_PROPERTY", "tgt": pid, "attrs": attrs})
                    edges.append({"src": paper_id, "rel": "MENTIONS", "tgt": pid, "attrs": {}})
                    if p.method_ref and p.method_ref in meth_name_to_id:
                        edges.append({"src": meth_name_to_id[p.method_ref], "rel": "REPORTS", "tgt": pid, "attrs": {}})

    logger.info(f"Graph construction completed: {len(nodes)} nodes, {len(edges)} edges")

    nodes_csv = DATA / "nodes.csv"
    edges_csv = DATA / "edges.csv"
    # with nodes_csv.open("w", newline="") as nf:
    with nodes_csv.open("w", newline="", encoding="utf-8") as nf:
        w = csv.writer(nf); w.writerow(["id:ID","type","props:JSON"])
        for n in nodes.values(): w.writerow([n["id"], n["type"], json.dumps(n["props"] or {})])
    # with edges_csv.open("w", newline="") as ef:
    with edges_csv.open("w", newline="", encoding="utf-8") as ef:
        w = csv.writer(ef); w.writerow([":START_ID","rel:TYPE",":END_ID","attrs:JSON"])
        for e in edges: w.writerow([e["src"], e["rel"], e["tgt"], json.dumps(e["attrs"] or {})])
    return {"nodes": str(nodes_csv), "edges": str(edges_csv)}

@mcp.tool()
def normalize_extraction(resource_uri: str) -> Dict[str, int]:
    """
    Load *_entities.json, normalize fields (units, aliases, compute VED), and re-save.
    """
    import json
    ents_path = _cache_path(resource_uri, "_entities.json")
    # data = json.loads(ents_path.read_text())
    data = json.loads(ents_path.read_text(encoding="utf-8"))
    ext = LPBFExtraction(**data)

    # Example: unit normalization, alias collapsing
    for mat in ext.materials:
        if mat.lpbf_process and mat.lpbf_process.energy_density_J_mm3 is None:
            # model_validator already tries; nothing else needed here for now
            pass
        if mat.properties:
            for prop in mat.properties:
                # alias: YS -> yield_strength
                if prop.name.lower() in {"ys", "yield", "0.2% proof"}:
                    prop.name = "yield_strength"
                # units: GPa -> MPa
                if prop.unit.lower() == "gpa":
                    prop.value *= 1000.0
                    prop.unit = "MPa"

    # ents_path.write_text(ext.model_dump_json(indent=2))
    ents_path.write_text(ext.model_dump_json(indent=2), encoding="utf-8")
    return {"materials": sum(1 for _ in ext.materials)}


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
