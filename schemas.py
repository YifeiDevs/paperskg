from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator


# ---- Core sub-objects --------------------------------------------------------

class CompositionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    element: str = Field(..., description="Chemical symbol, e.g., Fe, Cr, Ni")
    amount: float = Field(..., description="Numeric amount")
    unit: Literal["wt%", "at%", "ppm"] = Field(..., description="Composition unit")


class PowderSpec(BaseModel):
    model_config = ConfigDict(extra="allow")  # suppliers add lots of ad-hoc metadata
    supplier: Optional[str] = None
    lot: Optional[str] = None
    atomization: Optional[str] = Field(None, description="e.g., gas, plasma")
    size_d10_um: Optional[float] = None
    size_d50_um: Optional[float] = None
    size_d90_um: Optional[float] = None
    morphology: Optional[str] = None
    purity_pct: Optional[float] = None
    oxygen_ppm: Optional[float] = None
    nitrogen_ppm: Optional[float] = None


class LPBFProcess(BaseModel):
    model_config = ConfigDict(extra="allow")
    machine: Optional[str] = Field(None, description="e.g., EOS M290")
    laser_type: Optional[str] = Field(None, description="e.g., fiber, Yb:YAG")
    laser_power_W: Optional[float] = None
    scan_speed_mm_s: Optional[float] = None
    hatch_spacing_um: Optional[float] = None
    layer_thickness_um: Optional[float] = None
    preheat_temp_C: Optional[float] = None
    scan_strategy: Optional[str] = Field(None, description="e.g., stripe, chessboard, 67° rotate")
    atmosphere: Optional[str] = None
    oxygen_ppm: Optional[float] = None
    build_plate: Optional[str] = None
    recoater: Optional[str] = None
    relative_density_pct: Optional[float] = None
    energy_density_J_mm3: Optional[float] = Field(
        None, description="If reported; auto-computed if missing and inputs available"
    )
    part_orientation: Optional[str] = None
    geometry: Optional[str] = None

    @model_validator(mode="after")
    def compute_energy_density_if_missing(self) -> "LPBFProcess":
        """
        Compute Volumetric Energy Density (J/mm^3) if not provided and inputs exist:
            VED = power(W) / (scan_speed(mm/s) * hatch_spacing(mm) * layer_thickness(mm))
        Inputs expected in W, mm/s, μm -> conversion handled below.
        """
        if self.energy_density_J_mm3 is None:
            if self.laser_power_W and self.scan_speed_mm_s and self.hatch_spacing_um and self.layer_thickness_um:
                try:
                    hs_mm = self.hatch_spacing_um / 1000.0
                    lt_mm = self.layer_thickness_um / 1000.0
                    self.energy_density_J_mm3 = self.laser_power_W / (self.scan_speed_mm_s * hs_mm * lt_mm)
                except Exception:
                    # leave as None if anything odd
                    pass
        return self


class PostProcessStep(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Optional[str] = Field(None, description="HIP, stress-relief, solution, aging, anneal")
    temp_C: Optional[float] = None
    time_h: Optional[float] = None
    pressure_MPa: Optional[float] = None
    atmosphere: Optional[str] = None
    cooling: Optional[str] = None
    notes: Optional[str] = None


class ExperimentalMethod(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = Field(..., description="e.g., tensile, hardness, EBSD, XRD, CT, density (Archimedes)")
    standard: Optional[str] = Field(None, description="e.g., ASTM E8/E8M")
    instrument: Optional[str] = None
    params: Optional[Dict[str, Any]] = Field(
        default=None, description="Method-specific parameters (e.g., strain_rate_s, step_size_um)"
    )


class PropertyConditions(BaseModel):
    model_config = ConfigDict(extra="allow")
    temperature_C: Optional[float] = None
    strain_rate_s: Optional[float] = None
    direction: Optional[str] = Field(None, description="relative to build: vertical/horizontal")
    specimen_orientation: Optional[str] = None
    relative_density_pct: Optional[float] = None
    porosity_pct: Optional[float] = None
    surface_condition: Optional[str] = Field(None, description="as-built, machined, polished")
    cycle_R_ratio: Optional[float] = None
    frequency_Hz: Optional[float] = None


class PropertyMeasurement(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = Field(..., description="e.g., yield_strength, UTS, elongation, hardness_HV, density, porosity_pct")
    value: float
    unit: str
    conditions: Optional[PropertyConditions] = None
    method_ref: Optional[str] = Field(None, description="Reference to an ExperimentalMethod.name")
    provenance: Optional[Dict[str, Optional[str]]] = Field(
        default=None,
        description="Evidence pointers",
        examples=[{"page_span": "5-6", "figure": "Fig.3a", "table": "Table 2", "quote": "…"}],
    )


# ---- Material & Paper --------------------------------------------------------

class Material(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    name: str
    formula: Optional[str] = None
    material_system: Optional[str] = None
    composition: Optional[List[CompositionItem]] = None
    powder: Optional[PowderSpec] = None
    lpbf_process: Optional[LPBFProcess] = None
    post_processing: Optional[List[PostProcessStep]] = None
    experimental_methods: Optional[List[ExperimentalMethod]] = None
    properties: Optional[List[PropertyMeasurement]] = None


class PaperMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    title: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None


class LPBFExtraction(BaseModel):
    """
    Top-level container for one paper's extraction, aligned with the schema.
    """
    model_config = ConfigDict(extra="forbid")
    materials: List[Material]
    paper: Optional[PaperMetadata] = None
