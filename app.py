from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


Axis = Literal["x", "y"]
REBAR_DIAMETERS_MM = [12.0, 16.0, 20.0, 25.0, 28.0, 32.0]
SHEAR_REBAR_DIAMETERS_MM = [10.0, 12.0, 16.0, 20.0, 25.0]
REBAR_FY_BY_DIA_MPA = {
    10.0: 390.0,
    12.0: 390.0,
    16.0: 390.0,
    20.0: 390.0,
    25.0: 390.0,
    28.0: 390.0,
    32.0: 490.0,
}
PMM_METHOD = "PMM surface"
BIAXIAL_METHODS = ["Load contour", "Linear", PMM_METHOD]
DEAD_LOAD_FACTOR = 1.40
BACKFILL_EV_FACTOR = 1.35
AASHTO_EH_FACTOR = 1.50
AASHTO_EH_AT_REST_FACTOR = 1.35
AASHTO_SURCHARGE_FACTOR = 1.50
AASHTO_LIVE_SURCHARGE_FACTOR = 1.75
STRUCTURE_TYPES = ["Abutment with backfill", "Wall Pier / Pier wall without backfill"]
EARTH_LOAD_CODES = ["AASHTO LRFD Strength I", "EN / Eurocode DA1 auto governing", "Custom factors"]
EN_TRAFFIC_CATEGORIES = {
    "Road traffic (gamma_Q = 1.35)": 1.35,
    "Rail traffic (gamma_Q = 1.45)": 1.45,
    "Other variable action (gamma_Q = 1.50)": 1.50,
}

SOIL_PRESETS = {
    "Thailand general backfill": {
        "gamma_soil_kn_m3": 18.0,
        "phi_deg": 30.0,
        "note": "General compacted granular backfill when project soil data is not available.",
    },
    "Thailand conservative backfill": {
        "gamma_soil_kn_m3": 20.0,
        "phi_deg": 28.0,
        "note": "More conservative preliminary values for uncertain backfill.",
    },
    "Custom": {
        "gamma_soil_kn_m3": 18.0,
        "phi_deg": 30.0,
        "note": "Use project-specific geotechnical values.",
    },
}
Q_OTHER_PRESETS_KPA = {
    "None - traffic handled by live surcharge (0 kPa)": 0.0,
    "Roadway pavement / light permanent load (5 kPa)": 5.0,
    "Thailand general allowance (10 kPa)": 10.0,
    "Heavy storage or construction allowance (20 kPa)": 20.0,
    "Custom": None,
}
TRAFFIC_SURCHARGE_PRESETS = {
    "Road bridge - AASHTO auto h_eq": {"type": "road_auto", "q_kpa": None},
    "Railway light/passenger - q_LS 30 kPa": {"type": "q", "q_kpa": 30.0},
    "Railway mixed/general - q_LS 50 kPa": {"type": "q", "q_kpa": 50.0},
    "Railway heavy/conservative - q_LS 60 kPa": {"type": "q", "q_kpa": 60.0},
    "Custom h_eq": {"type": "h_eq", "q_kpa": None},
    "Custom q_LS": {"type": "q_custom", "q_kpa": None},
}
APPROACH_SLAB_GAP_PRESETS = [
    "0.0 m - full soil contact / no bridge action",
    "1.0 m - good compacted backfill",
    "2.0 m - moderate settlement or local void",
    "3.0 m - poor or uncertain backfill",
    "Full slab length - conservative no soil support",
    "Custom",
]
APPROACH_SLAB_LIVE_LOAD_PRESETS = {
    "None": {"type": "none", "q_kpa": 0.0, "q_line_kn_m": 0.0, "default_width_m": 0.0},
    "Road - AASHTO HL-93 lane load UDL only (3.1 kPa over 3 m lane)": {
        "type": "area",
        "q_kpa": 9.3 / 3.0,
        "q_line_kn_m": 0.0,
        "default_width_m": 3.0,
    },
    "Road - EN LM1 lane 1 UDL only (9.0 kPa over 3 m lane)": {
        "type": "area",
        "q_kpa": 9.0,
        "q_line_kn_m": 0.0,
        "default_width_m": 3.0,
    },
    "Road - custom uniform pressure": {"type": "area_custom", "q_kpa": 10.0, "q_line_kn_m": 0.0, "default_width_m": 3.0},
    "Rail - LM71 UDL only (80 kN/m per track)": {"type": "line", "q_kpa": 0.0, "q_line_kn_m": 80.0, "default_width_m": 0.0},
    "Rail - custom line load per track": {"type": "line_custom", "q_kpa": 0.0, "q_line_kn_m": 80.0, "default_width_m": 0.0},
}
APPROACH_SLAB_FACTOR_PRESETS = {
    "AASHTO LRFD Strength I": {"dc": 1.25, "dw": 1.50, "ll": 1.75},
    "EN / Eurocode STR/GEO DA1 C1": {"dc": 1.35, "dw": 1.35, "ll": 1.35},
    "EN / Eurocode STR/GEO DA1 C2": {"dc": 1.00, "dw": 1.00, "ll": 1.30},
    "Custom factors": {"dc": None, "dw": None, "ll": None},
}

ALPHA_GUIDANCE_MD = """
**Load contour alpha guide**

`alpha = 1.00` gives the linear biaxial interaction:
`Mx/Mnx + My/Mny <= 1.0`. This is conservative and matches the straight-line
moment interaction form used in AASHTO LRFD Article 5.6.4.5 for the applicable
low-axial-load case.

`alpha = 1.50` gives a rounded load contour:
`(Mx/Mnx)^alpha + (My/Mny)^alpha <= 1.0`. This is a common preliminary
Bresler/PCA-style approximation for biaxial flexure and is usually less
conservative than `alpha = 1.00`.

Code references to verify for final design: ACI 318-19 Chapter 21 and Chapter 22
(strength reduction factors and sectional strength for axial load with flexure);
AASHTO LRFD Bridge Design Specifications Article 5.6.4.5 (biaxial flexure).
ACI/AASHTO do not prescribe one universal `alpha = 1.50`; treat it as an
engineering approximation unless the project specification explicitly accepts it.
"""


def rebar_fy_mpa(diameter_mm: float) -> float:
    key = float(int(round(diameter_mm)))
    return REBAR_FY_BY_DIA_MPA[key]


def rebar_label(diameter_mm: float) -> str:
    return f"DB{int(diameter_mm)}"


@dataclass(frozen=True)
class BearingResultant:
    pu_kn: float
    vx_kn: float
    vy_kn: float
    mux_knm: float
    muy_knm: float
    torsion_z_knm: float

    @property
    def design_mux_knm(self) -> float:
        return abs(self.mux_knm)

    @property
    def design_muy_knm(self) -> float:
        return abs(self.muy_knm)


@dataclass(frozen=True)
class DeadLoadSummary:
    volume_m3: float
    service_dl_kn: float
    factor: float
    uls_pu_z_kn: float


@dataclass(frozen=True)
class BackfillVerticalLoadSummary:
    width_m: float
    height_m: float
    length_x_m: float
    volume_m3: float
    gamma_soil_kn_m3: float
    service_ev_kn: float
    factor: float
    uls_pu_z_kn: float
    centroid_y_mm: float
    uls_mux_knm: float


@dataclass(frozen=True)
class ApproachSlabReactionSummary:
    length_y_m: float
    width_x_m: float
    gap_m: float
    seat_length_m: float
    thickness_m: float
    superimposed_dead_kpa: float
    live_load_preset: str
    live_load_basis: str
    live_q_kpa: float
    live_line_load_kn_m: float
    live_loaded_width_m: float
    track_count: int
    point_load_kn: float
    point_load_y_m: float
    service_dc_kn: float
    service_dw_kn: float
    service_ll_uniform_kn: float
    service_ll_point_kn: float
    service_ll_kn: float
    dc_factor: float
    dw_factor: float
    ll_factor: float
    uls_dc_kn: float
    uls_dw_kn: float
    uls_ll_kn: float
    uls_pu_z_kn: float
    centroid_y_mm: float
    uls_mux_knm: float


@dataclass(frozen=True)
class EarthPressureFactors:
    load_code: str
    combination: str
    eh_factor: float
    q_other_factor: float
    live_factor: float
    gamma_tan_phi: float
    phi_design_deg: float


@dataclass(frozen=True)
class EarthPressureSummary:
    load_code: str
    combination: str
    height_m: float
    width_m: float
    gamma_soil_kn_m3: float
    phi_deg: float
    phi_design_deg: float
    pressure_mode: str
    pressure_coefficient: float
    eh_factor: float
    q_other_factor: float
    live_factor: float
    gamma_tan_phi: float
    q_other_kpa: float
    live_q_kpa: float
    live_h_eq_m: float
    direction: str
    service_soil_kn_per_m: float
    service_other_kn_per_m: float
    service_live_kn_per_m: float
    uls_soil_vy_kn: float
    uls_other_vy_kn: float
    uls_live_vy_kn: float
    uls_vy_kn: float
    uls_soil_mux_knm: float
    uls_other_mux_knm: float
    uls_live_mux_knm: float
    uls_mux_knm: float


@dataclass(frozen=True)
class StemDesignSummary:
    mu_kNm_per_m: float
    vu_kN_per_m: float
    effective_depth_mm: float
    bar_dia_mm: float
    fy_mpa: float
    phi_flexure: float
    as_flexure_mm2_per_m: float
    as_min_mm2_per_m: float
    as_required_mm2_per_m: float
    bar_spacing_mm: float
    status: str


@dataclass(frozen=True)
class CodeParameters:
    name: str
    phi_compression: float
    phi_flexure: float
    phi_method: Literal["strain", "aashto_axial"]
    axial_cap_factor: float
    eps_cu: float = 0.003


@dataclass(frozen=True)
class Bar:
    x_mm: float
    y_mm: float
    area_mm2: float


@dataclass(frozen=True)
class SectionCheck:
    width_x_mm: float
    depth_y_mm: float
    as_total_mm2: float
    rho_percent: float
    bar_count: int
    bar_dia_mm: float
    fy_mpa: float
    bars_x_face: int
    bars_y_face: int
    clear_spacing_x_mm: float
    clear_spacing_y_mm: float
    min_clear_spacing_mm: float
    center_spacing_x_mm: float
    center_spacing_y_mm: float
    max_center_spacing_mm: float
    max_spacing_advisory_mm: float
    spacing_status: str
    phi_pmax_kn: float
    phi_mnx_at_pu_knm: float
    phi_mny_at_pu_knm: float
    axial_ratio: float
    mux_ratio: float
    muy_ratio: float
    biaxial_ratio: float
    biaxial_linear_ratio: float
    load_contour_ratio: float
    load_contour_alpha: float
    pmm_ratio: float
    pmm_capacity_radius_knm: float
    biaxial_method: str
    governing_ratio: float
    status: str
    curve_x: list[dict[str, float]]
    curve_y: list[dict[str, float]]
    pmm_surface: dict[str, list[list[float]]]
    pmm_slice: list[dict[str, float]]
    bars: list[Bar]


@dataclass(frozen=True)
class ShearCheckSummary:
    name: str
    axis: str
    code_basis: str
    vu_kn: float
    bw_mm: float
    d_mm: float
    phi_v: float
    vc_kn: float
    phi_vc_kn: float
    vs_required_kn: float
    avs_strength_mm2_per_m: float
    avs_min_mm2_per_m: float
    avs_required_mm2_per_m: float
    tie_dia_mm: float
    tie_legs: int
    tie_area_mm2: float
    spacing_required_mm: float
    spacing_code_max_mm: float
    spacing_recommended_mm: float
    spacing_provided_mm: float
    avs_provided_mm2_per_m: float
    phi_vn_provided_kn: float
    utilization: float
    status: str
    notes: str


def default_code_parameters(code_name: str) -> CodeParameters:
    if "AASHTO" in code_name.upper():
        return CodeParameters(
            name="AASHTO LRFD style",
            phi_compression=0.75,
            phi_flexure=0.90,
            phi_method="aashto_axial",
            axial_cap_factor=0.80,
        )
    return CodeParameters(
        name="ACI 318 style",
        phi_compression=0.65,
        phi_flexure=0.90,
        phi_method="strain",
        axial_cap_factor=0.80,
    )


def beta1_aci(fc_mpa: float) -> float:
    if fc_mpa <= 28.0:
        return 0.85
    return max(0.65, 0.85 - 0.05 * ((fc_mpa - 28.0) / 7.0))


def bar_area_mm2(diameter_mm: float) -> float:
    return math.pi * diameter_mm**2 / 4.0


def _round_down_spacing_mm(value: float, increment: float = 25.0) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return math.inf
    return max(increment, math.floor(value / increment) * increment)


def shear_design_style_parameters(code_name: str, fc_mpa: float) -> dict[str, float | str]:
    if "AASHTO" in code_name.upper():
        return {
            "basis": "AASHTO LRFD style simplified shear: phi_v=0.90, beta=2.0 preliminary MCFT-style Vc",
            "phi_v": 0.90,
            "vc_coeff": 0.083 * 2.0,
            "min_sqrt_coeff": 0.083,
            "min_mpa_coeff": 0.35,
            "vs_limit_kn_coeff": 0.25 * max(float(fc_mpa), 0.0),
            "max_spacing_regular_factor": 0.80,
            "max_spacing_regular_cap_mm": 600.0,
            "max_spacing_high_factor": 0.40,
            "max_spacing_high_cap_mm": 300.0,
            "high_shear_ratio": 0.50,
        }
    return {
        "basis": "ACI 318 style simplified one-way shear: phi_v=0.75, Vc=0.17 sqrt(fc') bw d",
        "phi_v": 0.75,
        "vc_coeff": 0.17,
        "min_sqrt_coeff": 0.062,
        "min_mpa_coeff": 0.35,
        "vs_limit_kn_coeff": 0.66 * math.sqrt(max(float(fc_mpa), 0.0)),
        "max_spacing_regular_factor": 0.50,
        "max_spacing_regular_cap_mm": 600.0,
        "max_spacing_high_factor": 0.25,
        "max_spacing_high_cap_mm": 300.0,
        "high_shear_ratio": 0.50,
    }


def shear_check_summary(
    *,
    name: str,
    axis: str,
    vu_kn: float,
    bw_mm: float,
    d_mm: float,
    fc_mpa: float,
    fyt_mpa: float,
    code_name: str,
    tie_dia_mm: float,
    tie_legs: int,
    spacing_provided_mm: float,
) -> ShearCheckSummary:
    vu = abs(float(vu_kn))
    bw = max(float(bw_mm), 1.0)
    d = max(float(d_mm), 1.0)
    fc = max(float(fc_mpa), 0.0)
    fyt = max(float(fyt_mpa), 1.0)
    params = shear_design_style_parameters(code_name, fc)
    phi_v = float(params["phi_v"])
    sqrt_fc = math.sqrt(fc)
    vc = float(params["vc_coeff"]) * sqrt_fc * bw * d / 1000.0
    phi_vc = phi_v * vc
    vs_required = max(vu / max(phi_v, 1e-9) - vc, 0.0)
    avs_strength = vs_required * 1_000_000.0 / max(fyt * d, 1e-9)
    min_required = vu > 0.5 * phi_vc
    avs_min = 0.0
    if min_required:
        avs_min = max(
            float(params["min_sqrt_coeff"]) * sqrt_fc * bw / fyt,
            float(params["min_mpa_coeff"]) * bw / fyt,
        ) * 1000.0
    avs_required = max(avs_strength, avs_min)
    tie_area = max(int(tie_legs), 0) * bar_area_mm2(float(tie_dia_mm))
    spacing_required = tie_area * 1000.0 / avs_required if avs_required > 1e-9 else math.inf
    regular_max = min(float(params["max_spacing_regular_factor"]) * d, float(params["max_spacing_regular_cap_mm"]))
    high_max = min(float(params["max_spacing_high_factor"]) * d, float(params["max_spacing_high_cap_mm"]))
    spacing_code_max = high_max if vs_required > float(params["high_shear_ratio"]) * max(vc, 1e-9) else regular_max
    spacing_recommended = min(spacing_required, spacing_code_max)
    spacing_recommended = _round_down_spacing_mm(spacing_recommended)
    spacing_provided = max(float(spacing_provided_mm), 1.0)
    avs_provided = tie_area * 1000.0 / spacing_provided
    vs_provided = (avs_provided / 1000.0) * fyt * d / 1000.0
    phi_vn = phi_v * (vc + vs_provided)
    utilization = vu / max(phi_vn, 1e-9)

    nominal_limit = float(params["vs_limit_kn_coeff"]) * bw * d / 1000.0
    section_limit_ok = vs_required <= nominal_limit if "AASHTO" not in code_name.upper() else (vc + vs_required) <= nominal_limit
    spacing_ok = spacing_provided <= spacing_code_max + 1e-9
    strength_ok = vu <= phi_vn + 1e-9
    min_ok = (not min_required) or avs_provided >= avs_min - 1e-9
    status = "OK" if strength_ok and min_ok and spacing_ok and section_limit_ok else "NG"
    notes = []
    if min_required:
        notes.append("minimum shear reinforcement applies")
    else:
        notes.append("minimum shear reinforcement not triggered by Vu <= 0.5 phi Vc")
    if not section_limit_ok:
        notes.append("section shear limit exceeded; increase section size/strength")
    if not spacing_ok:
        notes.append("provided spacing exceeds code-style maximum")
    if "AASHTO" in code_name.upper():
        notes.append("AASHTO full shear design should verify theta/beta by current LRFD provisions")
    return ShearCheckSummary(
        name=name,
        axis=axis,
        code_basis=str(params["basis"]),
        vu_kn=vu,
        bw_mm=bw,
        d_mm=d,
        phi_v=phi_v,
        vc_kn=vc,
        phi_vc_kn=phi_vc,
        vs_required_kn=vs_required,
        avs_strength_mm2_per_m=avs_strength,
        avs_min_mm2_per_m=avs_min,
        avs_required_mm2_per_m=avs_required,
        tie_dia_mm=float(tie_dia_mm),
        tie_legs=int(tie_legs),
        tie_area_mm2=tie_area,
        spacing_required_mm=spacing_required,
        spacing_code_max_mm=spacing_code_max,
        spacing_recommended_mm=spacing_recommended,
        spacing_provided_mm=spacing_provided,
        avs_provided_mm2_per_m=avs_provided,
        phi_vn_provided_kn=phi_vn,
        utilization=utilization,
        status=status,
        notes="; ".join(notes),
    )


def clear_spacing_by_face_mm(
    width_x_mm: float,
    depth_y_mm: float,
    cover_mm: float,
    bar_dia_mm: float,
    bars_x_face: int,
    bars_y_face: int,
) -> tuple[float, float, float]:
    edge_x = width_x_mm / 2.0 - cover_mm - bar_dia_mm / 2.0
    edge_y = depth_y_mm / 2.0 - cover_mm - bar_dia_mm / 2.0
    if bars_x_face <= 1 or bars_y_face <= 1 or edge_x <= 0 or edge_y <= 0:
        return -math.inf, -math.inf, -math.inf
    clear_x = 2.0 * edge_x / (bars_x_face - 1) - bar_dia_mm
    clear_y = 2.0 * edge_y / (bars_y_face - 1) - bar_dia_mm
    return clear_x, clear_y, min(clear_x, clear_y)


def center_spacing_by_face_mm(
    width_x_mm: float,
    depth_y_mm: float,
    cover_mm: float,
    bar_dia_mm: float,
    bars_x_face: int,
    bars_y_face: int,
) -> tuple[float, float, float]:
    edge_x = width_x_mm / 2.0 - cover_mm - bar_dia_mm / 2.0
    edge_y = depth_y_mm / 2.0 - cover_mm - bar_dia_mm / 2.0
    if bars_x_face <= 1 or bars_y_face <= 1 or edge_x <= 0 or edge_y <= 0:
        return math.inf, math.inf, math.inf
    spacing_x = 2.0 * edge_x / (bars_x_face - 1)
    spacing_y = 2.0 * edge_y / (bars_y_face - 1)
    return spacing_x, spacing_y, max(spacing_x, spacing_y)


def layout_balance_penalty(
    width_x_mm: float,
    depth_y_mm: float,
    cover_mm: float,
    bar_dia_mm: float,
    bars_x_face: int,
    bars_y_face: int,
) -> float:
    edge_x = width_x_mm / 2.0 - cover_mm - bar_dia_mm / 2.0
    edge_y = depth_y_mm / 2.0 - cover_mm - bar_dia_mm / 2.0
    if bars_x_face <= 1 or bars_y_face <= 1 or edge_x <= 0 or edge_y <= 0:
        return math.inf
    spacing_x = 2.0 * edge_x / (bars_x_face - 1)
    spacing_y = 2.0 * edge_y / (bars_y_face - 1)
    return abs(math.log(max(spacing_x, 1.0) / max(spacing_y, 1.0)))


def combine_bearing_loads(records: Iterable[dict]) -> BearingResultant:
    pu = vx = vy = mux = muy = torsion = 0.0
    for row in records:
        x = float(row.get("x_mm", 0.0))
        y = float(row.get("y_mm", 0.0))
        z = float(row.get("z_mm", 0.0))
        px = float(row.get("Pu_x_kN", 0.0))
        py = float(row.get("Pu_y_kN", 0.0))
        pz = float(row.get("Pu_z_kN", 0.0))
        mx = float(row.get("Mu_x_kNm", 0.0))
        my = float(row.get("Mu_y_kNm", 0.0))

        pu += pz
        vx += px
        vy += py
        mux += mx + (-y * pz - z * py) / 1000.0
        muy += my + (z * px + x * pz) / 1000.0
        torsion += (x * py - y * px) / 1000.0

    return BearingResultant(
        pu_kn=pu,
        vx_kn=vx,
        vy_kn=vy,
        mux_knm=mux,
        muy_knm=muy,
        torsion_z_knm=torsion,
    )


def abutment_dead_load_summary(
    width_x_mm: float,
    depth_y_mm: float,
    height_z_mm: float,
    unit_weight_kn_m3: float,
    factor: float = DEAD_LOAD_FACTOR,
) -> DeadLoadSummary:
    volume_m3 = (
        max(float(width_x_mm), 0.0)
        * max(float(depth_y_mm), 0.0)
        * max(float(height_z_mm), 0.0)
        / 1_000_000_000.0
    )
    service_dl_kn = volume_m3 * max(float(unit_weight_kn_m3), 0.0)
    dead_load_factor = float(factor)
    return DeadLoadSummary(
        volume_m3=volume_m3,
        service_dl_kn=service_dl_kn,
        factor=dead_load_factor,
        uls_pu_z_kn=dead_load_factor * service_dl_kn,
    )


def backfill_vertical_load_summary(
    *,
    width_x_mm: float,
    depth_y_mm: float,
    backfill_width_m: float,
    backfill_height_m: float,
    gamma_soil_kn_m3: float,
    factor: float,
    earth_pressure_direction: str,
) -> BackfillVerticalLoadSummary:
    length_x_m = max(float(width_x_mm) / 1000.0, 0.0)
    width_m = max(float(backfill_width_m), 0.0)
    height_m = max(float(backfill_height_m), 0.0)
    gamma_soil = max(float(gamma_soil_kn_m3), 0.0)
    volume_m3 = length_x_m * width_m * height_m
    service_ev_kn = volume_m3 * gamma_soil
    load_factor = float(factor)
    uls_pu_z_kn = load_factor * service_ev_kn
    soil_side_sign = -1.0 if str(earth_pressure_direction).strip().startswith("+") else 1.0
    centroid_y_mm = soil_side_sign * (max(float(depth_y_mm), 0.0) / 2.0 + width_m * 1000.0 / 2.0)
    return BackfillVerticalLoadSummary(
        width_m=width_m,
        height_m=height_m,
        length_x_m=length_x_m,
        volume_m3=volume_m3,
        gamma_soil_kn_m3=gamma_soil,
        service_ev_kn=service_ev_kn,
        factor=load_factor,
        uls_pu_z_kn=uls_pu_z_kn,
        centroid_y_mm=centroid_y_mm,
        uls_mux_knm=-centroid_y_mm * uls_pu_z_kn / 1000.0,
    )


def approach_slab_gap_from_preset(preset: str, slab_length_m: float, custom_gap_m: float) -> float:
    if preset.startswith("0.0"):
        return 0.0
    if preset.startswith("1.0"):
        return min(1.0, max(float(slab_length_m), 0.0))
    if preset.startswith("2.0"):
        return min(2.0, max(float(slab_length_m), 0.0))
    if preset.startswith("3.0"):
        return min(3.0, max(float(slab_length_m), 0.0))
    if preset.startswith("Full slab length"):
        return max(float(slab_length_m), 0.0)
    return min(max(float(custom_gap_m), 0.0), max(float(slab_length_m), 0.0))


def approach_slab_reaction_summary(
    *,
    slab_length_y_m: float,
    slab_width_x_m: float,
    slab_thickness_m: float,
    seat_length_m: float,
    concrete_unit_weight_kn_m3: float,
    superimposed_dead_kpa: float,
    gap_m: float,
    live_load_preset: str,
    live_q_kpa: float,
    live_loaded_width_m: float,
    live_line_load_kn_m: float,
    track_count: int,
    point_load_kn: float,
    point_load_y_m: float,
    dc_factor: float,
    dw_factor: float,
    ll_factor: float,
    depth_y_mm: float,
    earth_pressure_direction: str,
) -> ApproachSlabReactionSummary:
    length_m = max(float(slab_length_y_m), 0.0)
    width_m = max(float(slab_width_x_m), 0.0)
    gap = min(max(float(gap_m), 0.0), length_m)
    thickness = max(float(slab_thickness_m), 0.0)
    seat_m = min(max(float(seat_length_m), 0.0), max(float(depth_y_mm), 0.0) / 1000.0)
    q_dc = max(float(concrete_unit_weight_kn_m3), 0.0) * thickness
    q_dw = max(float(superimposed_dead_kpa), 0.0)
    service_dc = 0.5 * q_dc * gap * width_m
    service_dw = 0.5 * q_dw * gap * width_m

    preset = APPROACH_SLAB_LIVE_LOAD_PRESETS.get(live_load_preset, APPROACH_SLAB_LIVE_LOAD_PRESETS["None"])
    live_type = str(preset["type"])
    loaded_width = min(max(float(live_loaded_width_m), 0.0), width_m)
    tracks = max(int(track_count), 0)
    q_ll = max(float(live_q_kpa), 0.0)
    line_ll = max(float(live_line_load_kn_m), 0.0)
    if live_type in {"area", "area_custom"}:
        service_ll_uniform = 0.5 * q_ll * gap * loaded_width
        live_basis = f"0.5 x q_LL x l_gap x loaded width ({loaded_width:.2f} m)"
    elif live_type in {"line", "line_custom"}:
        service_ll_uniform = 0.5 * line_ll * gap * tracks
        live_basis = f"0.5 x q_line x l_gap x {tracks} track(s)"
        loaded_width = 0.0
    else:
        service_ll_uniform = 0.0
        live_basis = "No approach slab live load"

    point_y = max(float(point_load_y_m), 0.0)
    point_load = max(float(point_load_kn), 0.0)
    if gap > 1e-9 and point_y <= gap:
        service_ll_point = point_load * (gap - point_y) / gap
    else:
        service_ll_point = 0.0

    service_ll = service_ll_uniform + service_ll_point
    uls_dc = max(float(dc_factor), 0.0) * service_dc
    uls_dw = max(float(dw_factor), 0.0) * service_dw
    uls_ll = max(float(ll_factor), 0.0) * service_ll
    uls_pu = uls_dc + uls_dw + uls_ll
    soil_side_sign = -1.0 if str(earth_pressure_direction).strip().startswith("+") else 1.0
    soil_face_y_mm = soil_side_sign * max(float(depth_y_mm), 0.0) / 2.0
    centroid_y_mm = soil_face_y_mm - soil_side_sign * seat_m * 1000.0 / 2.0
    return ApproachSlabReactionSummary(
        length_y_m=length_m,
        width_x_m=width_m,
        gap_m=gap,
        seat_length_m=seat_m,
        thickness_m=thickness,
        superimposed_dead_kpa=q_dw,
        live_load_preset=live_load_preset,
        live_load_basis=live_basis,
        live_q_kpa=q_ll,
        live_line_load_kn_m=line_ll,
        live_loaded_width_m=loaded_width,
        track_count=tracks,
        point_load_kn=point_load,
        point_load_y_m=point_y,
        service_dc_kn=service_dc,
        service_dw_kn=service_dw,
        service_ll_uniform_kn=service_ll_uniform,
        service_ll_point_kn=service_ll_point,
        service_ll_kn=service_ll,
        dc_factor=max(float(dc_factor), 0.0),
        dw_factor=max(float(dw_factor), 0.0),
        ll_factor=max(float(ll_factor), 0.0),
        uls_dc_kn=uls_dc,
        uls_dw_kn=uls_dw,
        uls_ll_kn=uls_ll,
        uls_pu_z_kn=uls_pu,
        centroid_y_mm=centroid_y_mm,
        uls_mux_knm=-centroid_y_mm * uls_pu / 1000.0,
    )


def earth_pressure_coefficient(phi_deg: float, pressure_mode: str) -> float:
    phi_rad = math.radians(min(max(float(phi_deg), 0.0), 45.0))
    sin_phi = math.sin(phi_rad)
    if pressure_mode.startswith("At-rest"):
        return max(0.0, 1.0 - sin_phi)
    return max(0.0, (1.0 - sin_phi) / max(1.0 + sin_phi, 1e-9))


def phi_design_from_gamma_tan(phi_deg: float, gamma_tan_phi: float) -> float:
    phi_rad = math.radians(min(max(float(phi_deg), 0.0), 45.0))
    gamma = max(float(gamma_tan_phi), 1e-9)
    return math.degrees(math.atan(math.tan(phi_rad) / gamma))


def aashto_road_h_eq_m(height_m: float) -> float:
    height = max(float(height_m), 0.0)
    if height <= 1.5:
        return 1.20
    if height <= 3.0:
        return 1.20 + (height - 1.5) * (0.90 - 1.20) / 1.5
    if height <= 6.0:
        return 0.90 + (height - 3.0) * (0.60 - 0.90) / 3.0
    return 0.60


def earth_pressure_factor_sets(
    load_code: str,
    pressure_mode: str,
    phi_deg: float,
    en_traffic_category: str,
    custom_eh_factor: float,
    custom_q_other_factor: float,
    custom_live_factor: float,
    custom_gamma_tan_phi: float,
) -> list[EarthPressureFactors]:
    if load_code.startswith("AASHTO"):
        eh_factor = AASHTO_EH_AT_REST_FACTOR if pressure_mode.startswith("At-rest") else AASHTO_EH_FACTOR
        return [
            EarthPressureFactors(
                load_code="AASHTO LRFD",
                combination="Strength I",
                eh_factor=eh_factor,
                q_other_factor=AASHTO_SURCHARGE_FACTOR,
                live_factor=AASHTO_LIVE_SURCHARGE_FACTOR,
                gamma_tan_phi=1.00,
                phi_design_deg=float(phi_deg),
            )
        ]

    if load_code.startswith("EN"):
        traffic_factor_c1 = EN_TRAFFIC_CATEGORIES.get(en_traffic_category, 1.35)
        return [
            EarthPressureFactors(
                load_code="EN / Eurocode",
                combination="DA1 C1: A1 + M1",
                eh_factor=1.35,
                q_other_factor=1.35,
                live_factor=traffic_factor_c1,
                gamma_tan_phi=1.00,
                phi_design_deg=float(phi_deg),
            ),
            EarthPressureFactors(
                load_code="EN / Eurocode",
                combination="DA1 C2: A2 + M2",
                eh_factor=1.00,
                q_other_factor=1.00,
                live_factor=1.30,
                gamma_tan_phi=1.25,
                phi_design_deg=phi_design_from_gamma_tan(phi_deg, 1.25),
            ),
        ]

    gamma_tan = max(float(custom_gamma_tan_phi), 1e-9)
    return [
        EarthPressureFactors(
            load_code="Custom factors",
            combination="User factors",
            eh_factor=float(custom_eh_factor),
            q_other_factor=float(custom_q_other_factor),
            live_factor=float(custom_live_factor),
            gamma_tan_phi=gamma_tan,
            phi_design_deg=phi_design_from_gamma_tan(phi_deg, gamma_tan),
        )
    ]


def earth_pressure_summary(
    width_x_mm: float,
    height_m: float,
    gamma_soil_kn_m3: float,
    phi_deg: float,
    pressure_mode: str,
    q_other_kpa: float,
    live_q_kpa: float,
    live_h_eq_m: float,
    direction: str,
    factors: EarthPressureFactors,
) -> EarthPressureSummary:
    height = max(float(height_m), 0.0)
    width_m = max(float(width_x_mm) / 1000.0, 1e-9)
    gamma_soil = max(float(gamma_soil_kn_m3), 0.0)
    k = earth_pressure_coefficient(factors.phi_design_deg, pressure_mode)
    q_other = max(float(q_other_kpa), 0.0)
    live_q = max(float(live_q_kpa), 0.0)
    sign = -1.0 if str(direction).strip().startswith("-") else 1.0

    soil_force_kn_per_m = 0.5 * k * gamma_soil * height**2
    other_force_kn_per_m = k * q_other * height
    live_force_kn_per_m = k * live_q * height

    soil_moment_knm_per_m = soil_force_kn_per_m * height / 3.0
    other_moment_knm_per_m = other_force_kn_per_m * height / 2.0
    live_moment_knm_per_m = live_force_kn_per_m * height / 2.0

    soil_vy = sign * factors.eh_factor * soil_force_kn_per_m * width_m
    other_vy = sign * factors.q_other_factor * other_force_kn_per_m * width_m
    live_vy = sign * factors.live_factor * live_force_kn_per_m * width_m

    soil_mux = -sign * factors.eh_factor * soil_moment_knm_per_m * width_m
    other_mux = -sign * factors.q_other_factor * other_moment_knm_per_m * width_m
    live_mux = -sign * factors.live_factor * live_moment_knm_per_m * width_m

    return EarthPressureSummary(
        load_code=factors.load_code,
        combination=factors.combination,
        height_m=height,
        width_m=width_m,
        gamma_soil_kn_m3=gamma_soil,
        phi_deg=float(phi_deg),
        phi_design_deg=factors.phi_design_deg,
        pressure_mode=pressure_mode,
        pressure_coefficient=k,
        eh_factor=factors.eh_factor,
        q_other_factor=factors.q_other_factor,
        live_factor=factors.live_factor,
        gamma_tan_phi=factors.gamma_tan_phi,
        q_other_kpa=q_other,
        live_q_kpa=live_q,
        live_h_eq_m=max(float(live_h_eq_m), 0.0),
        direction="+y" if sign > 0 else "-y",
        service_soil_kn_per_m=soil_force_kn_per_m,
        service_other_kn_per_m=other_force_kn_per_m,
        service_live_kn_per_m=live_force_kn_per_m,
        uls_soil_vy_kn=soil_vy,
        uls_other_vy_kn=other_vy,
        uls_live_vy_kn=live_vy,
        uls_vy_kn=soil_vy + other_vy + live_vy,
        uls_soil_mux_knm=soil_mux,
        uls_other_mux_knm=other_mux,
        uls_live_mux_knm=live_mux,
        uls_mux_knm=soil_mux + other_mux + live_mux,
    )


def governing_earth_pressure_summary(summaries: list[EarthPressureSummary]) -> EarthPressureSummary | None:
    if not summaries:
        return None
    return max(summaries, key=lambda summary: abs(summary.uls_mux_knm))


def earth_pressure_factor_table(summaries: list[EarthPressureSummary]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Code / combination": f"{summary.load_code} - {summary.combination}",
                "gamma_EH": f"{summary.eh_factor:.2f}",
                "gamma_q_other": f"{summary.q_other_factor:.2f}",
                "gamma_LS": f"{summary.live_factor:.2f}",
                "gamma_tan_phi": f"{summary.gamma_tan_phi:.2f}",
                "phi_design": f"{summary.phi_design_deg:.1f} deg",
                "K": f"{summary.pressure_coefficient:.3f}",
                "ULS Mu_x": f"{summary.uls_mux_knm:,.2f} kN-m",
            }
            for summary in summaries
        ]
    )


def stem_strip_design_summary(
    earth_pressure: EarthPressureSummary,
    depth_y_mm: float,
    cover_mm: float,
    bar_dia_mm: float,
    fc_mpa: float,
    fy_mpa: float,
    phi_flexure: float,
    min_rho: float = 0.0018,
) -> StemDesignSummary:
    width_strip_mm = 1000.0
    thickness_mm = max(float(depth_y_mm), 1.0)
    bar_dia = float(bar_dia_mm)
    effective_depth_mm = thickness_mm - max(float(cover_mm), 0.0) - bar_dia / 2.0
    mu_kNm_per_m = abs(earth_pressure.uls_mux_knm) / max(earth_pressure.width_m, 1e-9)
    vu_kN_per_m = abs(earth_pressure.uls_vy_kn) / max(earth_pressure.width_m, 1e-9)
    mu_nmm = mu_kNm_per_m * 1_000_000.0
    fc = max(float(fc_mpa), 1e-9)
    fy = max(float(fy_mpa), 1e-9)
    phi = max(float(phi_flexure), 1e-9)
    as_min = min_rho * width_strip_mm * thickness_mm
    status = "OK"
    as_flexure = math.inf
    if effective_depth_mm <= 0.0:
        status = "Invalid effective depth"
    else:
        a_coeff = fy**2 / (2.0 * 0.85 * fc * width_strip_mm)
        b_coeff = fy * effective_depth_mm
        c_coeff = mu_nmm / phi
        discriminant = b_coeff**2 - 4.0 * a_coeff * c_coeff
        if discriminant < 0.0:
            status = "Mu exceeds singly reinforced strip capacity"
        else:
            as_flexure = (b_coeff - math.sqrt(discriminant)) / (2.0 * a_coeff)
    as_required = max(as_min, as_flexure) if math.isfinite(as_flexure) else math.inf
    spacing_mm = bar_area_mm2(bar_dia) * 1000.0 / as_required if as_required > 0 and math.isfinite(as_required) else math.inf
    return StemDesignSummary(
        mu_kNm_per_m=mu_kNm_per_m,
        vu_kN_per_m=vu_kN_per_m,
        effective_depth_mm=effective_depth_mm,
        bar_dia_mm=bar_dia,
        fy_mpa=fy,
        phi_flexure=phi,
        as_flexure_mm2_per_m=as_flexure,
        as_min_mm2_per_m=as_min,
        as_required_mm2_per_m=as_required,
        bar_spacing_mm=spacing_mm,
        status=status,
    )


def pilecap_base_force_summary_table(
    resultant: BearingResultant,
    width_x_mm: float,
    dead_load: DeadLoadSummary | None = None,
    backfill_vertical: BackfillVerticalLoadSummary | None = None,
    approach_slab: ApproachSlabReactionSummary | None = None,
    earth_pressure: EarthPressureSummary | None = None,
) -> pd.DataFrame:
    width_m = max(float(width_x_mm) / 1000.0, 1e-9)
    dead_load_pu_kn = dead_load.uls_pu_z_kn if dead_load is not None else 0.0
    backfill_pu_kn = backfill_vertical.uls_pu_z_kn if backfill_vertical is not None else 0.0
    approach_slab_pu_kn = approach_slab.uls_pu_z_kn if approach_slab is not None else 0.0
    total_pu_kn = resultant.pu_kn + dead_load_pu_kn + backfill_pu_kn + approach_slab_pu_kn
    rows = [
        {
            "Resultant for pile cap": "Pu_z from bearings",
            "Point at centroid": f"{resultant.pu_kn:,.2f} kN",
            "Linearized over x width": f"{resultant.pu_kn / width_m:,.2f} kN/m",
            "Basis": "sum(Pu_z from bearing table)",
        },
    ]
    if dead_load is not None:
        rows.extend(
            [
                {
                    "Resultant for pile cap": "Abutment/pier self-weight DL",
                    "Point at centroid": f"{dead_load.service_dl_kn:,.2f} kN",
                    "Linearized over x width": f"{dead_load.service_dl_kn / width_m:,.2f} kN/m",
                    "Basis": "gamma_c x Lx x t x H",
                },
                {
                    "Resultant for pile cap": "Pu_z from 1.40D self-weight",
                    "Point at centroid": f"{dead_load.uls_pu_z_kn:,.2f} kN",
                    "Linearized over x width": f"{dead_load.uls_pu_z_kn / width_m:,.2f} kN/m",
                    "Basis": f"{dead_load.factor:.2f} x self-weight DL",
                },
            ]
        )
    if backfill_vertical is not None:
        rows.extend(
            [
                {
                    "Resultant for pile cap": "Backfill vertical load EV",
                    "Point at centroid": f"{backfill_vertical.service_ev_kn:,.2f} kN",
                    "Linearized over x width": f"{backfill_vertical.service_ev_kn / width_m:,.2f} kN/m",
                    "Basis": "gamma_soil x Lx x b_EV x h_EV",
                },
                {
                    "Resultant for pile cap": "Pu_z from factored EV",
                    "Point at centroid": f"{backfill_vertical.uls_pu_z_kn:,.2f} kN",
                    "Linearized over x width": f"{backfill_vertical.uls_pu_z_kn / width_m:,.2f} kN/m",
                    "Basis": f"{backfill_vertical.factor:.2f} x backfill EV",
                },
            ]
        )
    if approach_slab is not None:
        rows.extend(
            [
                {
                    "Resultant for pile cap": "Approach slab DC reaction",
                    "Point at centroid": f"{approach_slab.service_dc_kn:,.2f} kN",
                    "Linearized over x width": f"{approach_slab.service_dc_kn / width_m:,.2f} kN/m",
                    "Basis": "0.5 x gamma_c x t_slab x l_gap x B_slab",
                },
                {
                    "Resultant for pile cap": "Approach slab superimposed DW reaction",
                    "Point at centroid": f"{approach_slab.service_dw_kn:,.2f} kN",
                    "Linearized over x width": f"{approach_slab.service_dw_kn / width_m:,.2f} kN/m",
                    "Basis": "0.5 x q_DW x l_gap x B_slab",
                },
                {
                    "Resultant for pile cap": "Approach slab LL reaction",
                    "Point at centroid": f"{approach_slab.service_ll_kn:,.2f} kN",
                    "Linearized over x width": f"{approach_slab.service_ll_kn / width_m:,.2f} kN/m",
                    "Basis": approach_slab.live_load_basis,
                },
                {
                    "Resultant for pile cap": "Pu_z from factored approach slab",
                    "Point at centroid": f"{approach_slab.uls_pu_z_kn:,.2f} kN",
                    "Linearized over x width": f"{approach_slab.uls_pu_z_kn / width_m:,.2f} kN/m",
                    "Basis": (
                        f"{approach_slab.dc_factor:.2f}DC + {approach_slab.dw_factor:.2f}DW "
                        f"+ {approach_slab.ll_factor:.2f}LL"
                    ),
                },
            ]
        )
    pu_basis = ["bearing Pu_z"]
    if dead_load is not None:
        pu_basis.append("self-weight DL")
    if backfill_vertical is not None:
        pu_basis.append("backfill EV")
    if approach_slab is not None:
        pu_basis.append("approach slab reaction")
    rows.append(
        {
            "Resultant for pile cap": "Total Pu_z for pile cap ULS",
            "Point at centroid": f"{total_pu_kn:,.2f} kN",
            "Linearized over x width": f"{total_pu_kn / width_m:,.2f} kN/m",
            "Basis": " + ".join(pu_basis),
        }
    )
    if earth_pressure is not None:
        total_vy_kn = resultant.vy_kn + earth_pressure.uls_vy_kn
        rows.extend(
            [
                {
                    "Resultant for pile cap": "Vy from bearings",
                    "Point at centroid": f"{resultant.vy_kn:,.2f} kN",
                    "Linearized over x width": f"{resultant.vy_kn / width_m:,.2f} kN/m",
                    "Basis": "sum(Pu_y from bearing table)",
                },
                {
                    "Resultant for pile cap": "Vy from EH soil pressure",
                    "Point at centroid": f"{earth_pressure.uls_soil_vy_kn:,.2f} kN",
                    "Linearized over x width": f"{earth_pressure.uls_soil_vy_kn / width_m:,.2f} kN/m",
                    "Basis": f"{earth_pressure.eh_factor:.2f} x 0.5 x K x gamma_soil x H^2 x Lx",
                },
                {
                    "Resultant for pile cap": "Vy from q_other surcharge",
                    "Point at centroid": f"{earth_pressure.uls_other_vy_kn:,.2f} kN",
                    "Linearized over x width": f"{earth_pressure.uls_other_vy_kn / width_m:,.2f} kN/m",
                    "Basis": f"{earth_pressure.q_other_factor:.2f} x K x q_other x H x Lx",
                },
                {
                    "Resultant for pile cap": "Vy from traffic LS surcharge",
                    "Point at centroid": f"{earth_pressure.uls_live_vy_kn:,.2f} kN",
                    "Linearized over x width": f"{earth_pressure.uls_live_vy_kn / width_m:,.2f} kN/m",
                    "Basis": f"{earth_pressure.live_factor:.2f} x K x q_LS x H x Lx",
                },
                {
                    "Resultant for pile cap": "Total Vy for pile cap ULS",
                    "Point at centroid": f"{total_vy_kn:,.2f} kN",
                    "Linearized over x width": f"{total_vy_kn / width_m:,.2f} kN/m",
                    "Basis": "bearing Vy + factored EH/q_other/LS",
                },
            ]
        )
    rows.extend(
        [
            {
                "Resultant for pile cap": "Mu_x from bearings",
                "Point at centroid": f"{resultant.mux_knm:,.2f} kN-m",
                "Linearized over x width": f"{resultant.mux_knm / width_m:,.2f} kN-m/m",
                "Basis": "sum(Mu_x + (-y Pu_z - z Pu_y) / 1000)",
            },
        ]
    )
    if earth_pressure is not None:
        rows.append(
            {
                "Resultant for pile cap": "Mu_x from earth pressure ULS",
                "Point at centroid": f"{earth_pressure.uls_mux_knm:,.2f} kN-m",
                "Linearized over x width": f"{earth_pressure.uls_mux_knm / width_m:,.2f} kN-m/m",
                "Basis": "EH at H/3, q_other and LS at H/2",
            }
        )
    if backfill_vertical is not None:
        rows.append(
            {
                "Resultant for pile cap": "Mu_x from backfill EV eccentricity",
                "Point at centroid": f"{backfill_vertical.uls_mux_knm:,.2f} kN-m",
                "Linearized over x width": f"{backfill_vertical.uls_mux_knm / width_m:,.2f} kN-m/m",
                "Basis": f"-y_EV x Pu_EV / 1000, y_EV = {backfill_vertical.centroid_y_mm:,.0f} mm",
            }
        )
    if approach_slab is not None:
        rows.append(
            {
                "Resultant for pile cap": "Mu_x from approach slab reaction",
                "Point at centroid": f"{approach_slab.uls_mux_knm:,.2f} kN-m",
                "Linearized over x width": f"{approach_slab.uls_mux_knm / width_m:,.2f} kN-m/m",
                "Basis": (
                    f"-y_AS x Pu_AS / 1000, y_AS = {approach_slab.centroid_y_mm:,.0f} mm "
                    f"(seat = {approach_slab.seat_length_m:.2f} m)"
                ),
            }
        )
    if earth_pressure is not None or backfill_vertical is not None or approach_slab is not None:
        total_mux_knm = resultant.mux_knm
        mux_basis = ["bearing Mu_x"]
        if earth_pressure is not None:
            total_mux_knm += earth_pressure.uls_mux_knm
            mux_basis.append("earth pressure Mu_x")
        if backfill_vertical is not None:
            total_mux_knm += backfill_vertical.uls_mux_knm
            mux_basis.append("backfill EV eccentricity")
        if approach_slab is not None:
            total_mux_knm += approach_slab.uls_mux_knm
            mux_basis.append("approach slab eccentricity")
        rows.append(
            {
                "Resultant for pile cap": "Total Mu_x for pile cap ULS",
                "Point at centroid": f"{total_mux_knm:,.2f} kN-m",
                "Linearized over x width": f"{total_mux_knm / width_m:,.2f} kN-m/m",
                "Basis": " + ".join(mux_basis),
            }
        )
    rows.extend(
        [
            {
                "Resultant for pile cap": "Mu_y from bearings",
                "Point at centroid": f"{resultant.muy_knm:,.2f} kN-m",
                "Linearized over x width": "Use point couple",
                "Basis": "sum(Mu_y + (z Pu_x + x Pu_z) / 1000)",
            },
        ]
    )
    return pd.DataFrame(rows)


def section_design_resultant_with_earth_pressure(
    bearing_resultant: BearingResultant,
    earth_pressure: EarthPressureSummary | None,
    design_width_x_mm: float,
) -> tuple[BearingResultant, float, float, float]:
    if earth_pressure is None:
        return bearing_resultant, 0.0, 0.0, 0.0

    design_width_m = max(float(design_width_x_mm) / 1000.0, 0.0)
    earth_width_m = max(float(earth_pressure.width_m), 1e-9)
    width_ratio = min(1.0, design_width_m / earth_width_m)
    earth_vy_kn = earth_pressure.uls_vy_kn * width_ratio
    earth_mux_knm = earth_pressure.uls_mux_knm * width_ratio
    return (
        BearingResultant(
            pu_kn=bearing_resultant.pu_kn,
            vx_kn=bearing_resultant.vx_kn,
            vy_kn=bearing_resultant.vy_kn + earth_vy_kn,
            mux_knm=bearing_resultant.mux_knm + earth_mux_knm,
            muy_knm=bearing_resultant.muy_knm,
            torsion_z_knm=bearing_resultant.torsion_z_knm,
        ),
        width_ratio,
        earth_vy_kn,
        earth_mux_knm,
    )


def _min_positive_spacing_mm(values: Iterable[float]) -> float:
    unique_values = sorted({round(float(value), 3) for value in values})
    spacings = [
        unique_values[index + 1] - unique_values[index]
        for index in range(len(unique_values) - 1)
        if unique_values[index + 1] - unique_values[index] > 1e-6
    ]
    return min(spacings) if spacings else math.inf


def default_strength_bearing_names(records: list[dict]) -> list[str]:
    if not records:
        return []
    groups: dict[float, list[dict]] = {}
    for row in records:
        groups.setdefault(round(float(row.get("x_mm", 0.0)), 3), []).append(row)

    def group_score(group: list[dict]) -> float:
        return sum(
            abs(float(row.get("Pu_z_kN", 0.0)))
            + 0.25 * abs(float(row.get("Pu_x_kN", 0.0)))
            + 0.25 * abs(float(row.get("Pu_y_kN", 0.0)))
            + 0.05 * abs(float(row.get("Mu_x_kNm", 0.0)))
            + 0.05 * abs(float(row.get("Mu_y_kNm", 0.0)))
            for row in group
        )

    critical_group = max(groups.values(), key=group_score)
    return [str(row.get("name", "")) for row in critical_group]


def selected_bearings(records: list[dict], selected_names: Iterable[str]) -> list[dict]:
    selected = {str(name) for name in selected_names}
    return [row for row in records if str(row.get("name", "")) in selected]


def strip_center_x_mm(records: list[dict]) -> float:
    if not records:
        return 0.0
    weighted_sum = 0.0
    weight_total = 0.0
    for row in records:
        x = float(row.get("x_mm", 0.0))
        weight = abs(float(row.get("Pu_z_kN", 0.0)))
        weighted_sum += weight * x
        weight_total += weight
    if weight_total > 1e-9:
        return weighted_sum / weight_total
    return sum(float(row.get("x_mm", 0.0)) for row in records) / len(records)


def effective_strip_recommendation(
    *,
    records: list[dict],
    selected_names: Iterable[str],
    full_width_x_mm: float,
    depth_y_mm: float,
    bearing_size_x_mm: float,
) -> dict[str, float]:
    selected = selected_bearings(records, selected_names) or records
    selected_x = [float(row.get("x_mm", 0.0)) for row in selected]
    selected_min_x = min(selected_x) if selected_x else 0.0
    selected_max_x = max(selected_x) if selected_x else 0.0
    loaded_width = bearing_size_x_mm + (selected_max_x - selected_min_x if selected_x else 0.0)
    distribution_limit = loaded_width + 4.0 * depth_y_mm
    all_x = sorted({round(float(row.get("x_mm", 0.0)), 3) for row in records})
    spacing_limit = _min_positive_spacing_mm(all_x)
    if not math.isfinite(spacing_limit):
        spacing_limit = full_width_x_mm
    left_edge_x = -full_width_x_mm / 2.0
    right_edge_x = full_width_x_mm / 2.0
    left_edge_center_distance = max(0.0, selected_min_x - left_edge_x)
    right_edge_center_distance = max(0.0, right_edge_x - selected_max_x)
    left_neighbors = [x for x in all_x if x < selected_min_x - 1e-6]
    right_neighbors = [x for x in all_x if x > selected_max_x + 1e-6]
    left_boundary = (
        (max(left_neighbors) + selected_min_x) / 2.0
        if left_neighbors
        else left_edge_x
    )
    right_boundary = (
        (min(right_neighbors) + selected_max_x) / 2.0
        if right_neighbors
        else right_edge_x
    )
    tributary_limit = max(loaded_width, right_boundary - left_boundary)
    available_limit = full_width_x_mm
    recommended = min(distribution_limit, tributary_limit, available_limit)
    return {
        "loaded_width_mm": loaded_width,
        "distribution_limit_mm": distribution_limit,
        "spacing_limit_mm": spacing_limit,
        "left_edge_center_distance_mm": left_edge_center_distance,
        "right_edge_center_distance_mm": right_edge_center_distance,
        "tributary_limit_mm": tributary_limit,
        "available_limit_mm": available_limit,
        "recommended_width_mm": max(1.0, recommended),
    }


def localize_bearing_records(records: list[dict], strip_center_x: float) -> list[dict]:
    localized: list[dict] = []
    for row in records:
        local = dict(row)
        local["x_mm"] = float(row.get("x_mm", 0.0)) - strip_center_x
        localized.append(local)
    return localized


def make_perimeter_bars(
    width_x_mm: float,
    depth_y_mm: float,
    cover_mm: float,
    bar_dia_mm: float,
    bars_x_face: int,
    bars_y_face: int,
) -> list[Bar]:
    if bars_x_face < 2:
        raise ValueError("bars_x_face must be at least 2.")
    if bars_y_face < 2:
        raise ValueError("bars_y_face must be at least 2.")

    edge_x = width_x_mm / 2.0 - cover_mm - bar_dia_mm / 2.0
    edge_y = depth_y_mm / 2.0 - cover_mm - bar_dia_mm / 2.0
    if edge_x <= 0 or edge_y <= 0:
        raise ValueError("Cover and bar diameter do not fit inside the section.")

    area = bar_area_mm2(bar_dia_mm)
    points: list[tuple[float, float]] = []
    for x in np.linspace(-edge_x, edge_x, bars_x_face):
        points.append((float(x), edge_y))
        points.append((float(x), -edge_y))
    y_values = np.linspace(-edge_y, edge_y, bars_y_face)
    for y in y_values[1:-1]:
        points.append((-edge_x, float(y)))
        points.append((edge_x, float(y)))

    deduped: dict[tuple[int, int], Bar] = {}
    for x, y in points:
        deduped[(round(x), round(y))] = Bar(x_mm=x, y_mm=y, area_mm2=area)
    return list(deduped.values())


def _rectangle_polygon(width_x_mm: float, depth_y_mm: float) -> list[tuple[float, float]]:
    x = width_x_mm / 2.0
    y = depth_y_mm / 2.0
    return [(-x, -y), (x, -y), (x, y), (-x, y)]


def _clip_polygon_ge(
    polygon: list[tuple[float, float]],
    normal: tuple[float, float],
    threshold: float,
) -> list[tuple[float, float]]:
    if not polygon:
        return []

    nx, ny = normal

    def value(point: tuple[float, float]) -> float:
        return point[0] * nx + point[1] * ny

    def inside(point: tuple[float, float]) -> bool:
        return value(point) >= threshold - 1e-9

    output: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    previous_value = value(previous)

    for current in polygon:
        current_inside = inside(current)
        current_value = value(current)
        if current_inside != previous_inside:
            denom = current_value - previous_value
            if abs(denom) > 1e-12:
                t = (threshold - previous_value) / denom
                ix = previous[0] + t * (current[0] - previous[0])
                iy = previous[1] + t * (current[1] - previous[1])
                output.append((ix, iy))
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
        previous_value = current_value

    return output


def _polygon_area_centroid(polygon: list[tuple[float, float]]) -> tuple[float, float, float]:
    if len(polygon) < 3:
        return 0.0, 0.0, 0.0

    twice_area = 0.0
    cx_term = 0.0
    cy_term = 0.0
    for index, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(index + 1) % len(polygon)]
        cross = x0 * y1 - x1 * y0
        twice_area += cross
        cx_term += (x0 + x1) * cross
        cy_term += (y0 + y1) * cross

    area = twice_area / 2.0
    if abs(area) < 1e-9:
        return 0.0, 0.0, 0.0
    cx = cx_term / (6.0 * area)
    cy = cy_term / (6.0 * area)
    return abs(area), cx, cy


def _steel_stress_mpa(strain: float, fy_mpa: float, es_mpa: float) -> float:
    return max(-fy_mpa, min(fy_mpa, es_mpa * strain))


def _phi_factor(
    eps_t: float,
    eps_y: float,
    pn_n: float,
    fc_mpa: float,
    ag_mm2: float,
    params: CodeParameters,
) -> float:
    if params.phi_method == "aashto_axial":
        if pn_n <= 0:
            return params.phi_flexure
        threshold_n = max(1.0, 0.10 * fc_mpa * ag_mm2)
        ratio = min(1.0, max(0.0, pn_n / threshold_n))
        return params.phi_flexure - (params.phi_flexure - params.phi_compression) * ratio

    if eps_t <= eps_y:
        return params.phi_compression
    if eps_t >= eps_y + 0.003:
        return params.phi_flexure
    ratio = (eps_t - eps_y) / 0.003
    return params.phi_compression + (params.phi_flexure - params.phi_compression) * ratio


def _section_response(
    *,
    width_x_mm: float,
    depth_y_mm: float,
    bars: list[Bar],
    fc_mpa: float,
    fy_mpa: float,
    es_mpa: float,
    c_mm: float,
    theta_rad: float,
    params: CodeParameters,
) -> dict[str, float]:
    beta1 = beta1_aci(fc_mpa)
    normal = (math.cos(theta_rad), math.sin(theta_rad))
    polygon = _rectangle_polygon(width_x_mm, depth_y_mm)
    pmax = max(x * normal[0] + y * normal[1] for x, y in polygon)
    threshold = pmax - beta1 * c_mm
    compression_polygon = _clip_polygon_ge(polygon, normal, threshold)
    concrete_area, cx, cy = _polygon_area_centroid(compression_polygon)

    pn_n = 0.85 * fc_mpa * concrete_area
    mx_nmm = pn_n * cy
    my_nmm = -pn_n * cx
    min_strain = params.eps_cu

    for bar in bars:
        projection = bar.x_mm * normal[0] + bar.y_mm * normal[1]
        distance_from_compression_edge = pmax - projection
        strain = params.eps_cu * (1.0 - distance_from_compression_edge / c_mm)
        stress = _steel_stress_mpa(strain, fy_mpa, es_mpa)
        stress_block_contains_bar = projection >= threshold - 1e-9
        concrete_replacement_stress = 0.85 * fc_mpa if stress_block_contains_bar and stress > 0.0 else 0.0
        force_n = (stress - concrete_replacement_stress) * bar.area_mm2
        pn_n += force_n
        mx_nmm += force_n * bar.y_mm
        my_nmm += -force_n * bar.x_mm
        min_strain = min(min_strain, strain)

    eps_t = max(0.0, -min_strain)
    phi = _phi_factor(
        eps_t=eps_t,
        eps_y=fy_mpa / es_mpa,
        pn_n=pn_n,
        fc_mpa=fc_mpa,
        ag_mm2=width_x_mm * depth_y_mm,
        params=params,
    )
    return {
        "pn_kn": pn_n / 1000.0,
        "mx_knm": mx_nmm / 1_000_000.0,
        "my_knm": my_nmm / 1_000_000.0,
        "eps_t": eps_t,
        "phi": phi,
        "phi_pn_kn": phi * pn_n / 1000.0,
        "phi_mx_knm": phi * mx_nmm / 1_000_000.0,
        "phi_my_knm": phi * my_nmm / 1_000_000.0,
    }


def _phi_pmax_kn(
    width_x_mm: float,
    depth_y_mm: float,
    bars: list[Bar],
    fc_mpa: float,
    fy_mpa: float,
    params: CodeParameters,
) -> float:
    ag = width_x_mm * depth_y_mm
    ast = sum(bar.area_mm2 for bar in bars)
    po_n = 0.85 * fc_mpa * max(0.0, ag - ast) + fy_mpa * ast
    return params.axial_cap_factor * params.phi_compression * po_n / 1000.0


def shrinkage_temperature_ratio(code_name: str, fy_mpa: float) -> tuple[float, str]:
    if "AASHTO" in code_name.upper():
        # AASHTO style As >= 0.11 Ag / fy with fy in ksi; 0.11 * 6.89476 = 0.75842 for MPa.
        return 0.75842 / max(fy_mpa, 1.0), "AASHTO LRFD style: As >= 0.11 Ag / fy"
    if fy_mpa < 420.0:
        return 0.0020, "ACI shrinkage-temperature style: rho = 0.0020 for fy < 420 MPa"
    return max(0.0014, 0.0018 * 420.0 / fy_mpa), "ACI shrinkage-temperature style: rho >= 0.0018*420/fy, not less than 0.0014"


def minimum_reinforcement_check(check: SectionCheck, code_name: str) -> dict[str, float | str]:
    rho_req, basis = shrinkage_temperature_ratio(code_name, check.fy_mpa)
    ag = check.width_x_mm * check.depth_y_mm
    as_min_total = rho_req * ag
    bar_area = bar_area_mm2(check.bar_dia_mm)
    as_top_bottom_face = check.bars_x_face * bar_area
    as_min_each_main_face = as_min_total / 2.0
    total_status = "OK" if check.as_total_mm2 + 1e-9 >= as_min_total else "NG"
    face_status = "OK" if as_top_bottom_face + 1e-9 >= as_min_each_main_face else "NG"
    return {
        "basis": basis,
        "rho_req_percent": rho_req * 100.0,
        "as_min_total_mm2": as_min_total,
        "as_provided_total_mm2": check.as_total_mm2,
        "total_status": total_status,
        "as_min_each_main_face_mm2": as_min_each_main_face,
        "as_provided_each_top_bottom_face_mm2": as_top_bottom_face,
        "face_status": face_status,
    }


def minimum_reinforcement_box_html(check: SectionCheck, code_name: str) -> str:
    min_reinf = minimum_reinforcement_check(check, code_name)
    status_ok = min_reinf["total_status"] == "OK" and min_reinf["face_status"] == "OK"
    status_color = "#0f766e" if status_ok else "#b91c1c"
    status_label = "OK" if status_ok else "NG"
    return f"""
    <div style="border:1px solid #cbd5e1; border-radius:8px; padding:12px 14px; background:#f8fafc; margin-top:8px;">
      <div style="font-weight:700; color:#172033; margin-bottom:6px;">Minimum Reinforcement Check</div>
      <div style="font-size:0.92rem; color:#334155; line-height:1.55;">
        Shrinkage/temperature distributed reinforcement only. Column longitudinal minimum such as 1%Ag is not applied.<br>
        Required rho = <b>{float(min_reinf['rho_req_percent']):.3f}%</b><br>
        As,min total = <b>{float(min_reinf['as_min_total_mm2']):,.0f} mm2</b>,
        As provided total = <b>{float(min_reinf['as_provided_total_mm2']):,.0f} mm2</b>
        (<span style="color:{status_color}; font-weight:700;">{min_reinf['total_status']}</span>)<br>
        As,min each main face = <b>{float(min_reinf['as_min_each_main_face_mm2']):,.0f} mm2</b>,
        As provided each top/bottom face = <b>{float(min_reinf['as_provided_each_top_bottom_face_mm2']):,.0f} mm2</b>
        (<span style="color:{status_color}; font-weight:700;">{min_reinf['face_status']}</span>)<br>
        Overall status: <span style="color:{status_color}; font-weight:700;">{status_label}</span>
      </div>
    </div>
    """


def interaction_curve(
    *,
    width_x_mm: float,
    depth_y_mm: float,
    bars: list[Bar],
    fc_mpa: float,
    fy_mpa: float,
    es_mpa: float,
    params: CodeParameters,
    axis: Axis,
    sample_count: int = 180,
) -> list[dict[str, float]]:
    theta = math.pi / 2.0 if axis == "x" else math.pi
    max_dim = max(width_x_mm, depth_y_mm)
    c_values = np.geomspace(max(1.0, max_dim / 1000.0), max_dim * 80.0, sample_count)
    pmax = _phi_pmax_kn(width_x_mm, depth_y_mm, bars, fc_mpa, fy_mpa, params)

    curve: list[dict[str, float]] = []
    for c_mm in c_values:
        response = _section_response(
            width_x_mm=width_x_mm,
            depth_y_mm=depth_y_mm,
            bars=bars,
            fc_mpa=fc_mpa,
            fy_mpa=fy_mpa,
            es_mpa=es_mpa,
            c_mm=float(c_mm),
            theta_rad=theta,
            params=params,
        )
        phi_pn = min(response["phi_pn_kn"], pmax)
        moment_key = "phi_mx_knm" if axis == "x" else "phi_my_knm"
        curve.append(
            {
                "phi_pn_kn": phi_pn,
                "phi_mn_knm": abs(response[moment_key]),
                "phi": response["phi"],
                "eps_t": response["eps_t"],
            }
        )

    curve.append({"phi_pn_kn": pmax, "phi_mn_knm": 0.0, "phi": params.phi_compression, "eps_t": 0.0})
    return curve


def _capacity_at_pu(curve: list[dict[str, float]], pu_kn: float) -> float:
    by_p: dict[float, float] = {}
    for point in curve:
        p = round(point["phi_pn_kn"], 6)
        m = point["phi_mn_knm"]
        by_p[p] = max(by_p.get(p, 0.0), m)

    p_values = np.array(sorted(by_p.keys()), dtype=float)
    m_values = np.array([by_p[p] for p in p_values], dtype=float)
    if len(p_values) < 2:
        return 0.0
    if pu_kn > float(p_values.max()):
        return 0.0
    if pu_kn < float(p_values.min()):
        return float(m_values[0])
    return float(np.interp(pu_kn, p_values, m_values))


def _cross_2d(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _ray_polygon_capacity_radius(points: list[tuple[float, float]], mux_knm: float, muy_knm: float) -> float:
    demand_radius = math.hypot(mux_knm, muy_knm)
    if demand_radius <= 1e-9:
        return math.inf
    if len(points) < 3:
        return 0.0

    ux = mux_knm / demand_radius
    uy = muy_knm / demand_radius
    intersections: list[float] = []
    closed = points + [points[0]]
    for p1, p2 in zip(closed[:-1], closed[1:]):
        x1, y1 = p1
        dx = p2[0] - x1
        dy = p2[1] - y1
        denom = _cross_2d(ux, uy, dx, dy)
        if abs(denom) < 1e-9:
            continue
        t = _cross_2d(x1, y1, dx, dy) / denom
        s = _cross_2d(x1, y1, ux, uy) / denom
        if t >= -1e-7 and -1e-7 <= s <= 1.0 + 1e-7:
            intersections.append(max(0.0, t))

    positive = [value for value in intersections if value > 1e-6]
    return min(positive) if positive else 0.0


def pmm_surface_check(
    *,
    width_x_mm: float,
    depth_y_mm: float,
    bars: list[Bar],
    fc_mpa: float,
    fy_mpa: float,
    es_mpa: float,
    pu_kn: float,
    mux_knm: float,
    muy_knm: float,
    params: CodeParameters,
    angle_count: int = 72,
    sample_count: int = 84,
) -> dict:
    pmax = _phi_pmax_kn(width_x_mm, depth_y_mm, bars, fc_mpa, fy_mpa, params)
    max_dim = max(width_x_mm, depth_y_mm)
    c_values = np.geomspace(max(1.0, max_dim / 1000.0), max_dim * 80.0, sample_count)
    theta_values = np.linspace(0.0, 2.0 * math.pi, angle_count, endpoint=False)

    mx_grid: list[list[float]] = []
    my_grid: list[list[float]] = []
    p_grid: list[list[float]] = []
    slice_points: list[dict[str, float]] = []

    for theta in theta_values:
        curve: list[dict[str, float]] = []
        row_mx: list[float] = []
        row_my: list[float] = []
        row_p: list[float] = []
        for c_mm in c_values:
            response = _section_response(
                width_x_mm=width_x_mm,
                depth_y_mm=depth_y_mm,
                bars=bars,
                fc_mpa=fc_mpa,
                fy_mpa=fy_mpa,
                es_mpa=es_mpa,
                c_mm=float(c_mm),
                theta_rad=float(theta),
                params=params,
            )
            phi_pn = min(response["phi_pn_kn"], pmax)
            mx = response["phi_mx_knm"]
            my = response["phi_my_knm"]
            point = {"theta": float(theta), "phi_pn_kn": phi_pn, "phi_mx_knm": mx, "phi_my_knm": my}
            curve.append(point)
            row_mx.append(mx)
            row_my.append(my)
            row_p.append(phi_pn)

        mx_grid.append(row_mx)
        my_grid.append(row_my)
        p_grid.append(row_p)

        candidates: list[dict[str, float]] = []
        for p1, p2 in zip(curve[:-1], curve[1:]):
            p1_value = p1["phi_pn_kn"]
            p2_value = p2["phi_pn_kn"]
            if abs(p2_value - p1_value) < 1e-9:
                continue
            if (p1_value - pu_kn) * (p2_value - pu_kn) <= 0.0:
                ratio = (pu_kn - p1_value) / (p2_value - p1_value)
                ratio = min(1.0, max(0.0, ratio))
                mx = p1["phi_mx_knm"] + ratio * (p2["phi_mx_knm"] - p1["phi_mx_knm"])
                my = p1["phi_my_knm"] + ratio * (p2["phi_my_knm"] - p1["phi_my_knm"])
                candidates.append({"mx_knm": mx, "my_knm": my, "theta": float(theta), "angle": math.atan2(my, mx)})

        if candidates:
            slice_points.append(max(candidates, key=lambda point: math.hypot(point["mx_knm"], point["my_knm"])))

    if mx_grid:
        mx_grid.append(list(mx_grid[0]))
        my_grid.append(list(my_grid[0]))
        p_grid.append(list(p_grid[0]))

    if not slice_points and abs(pu_kn - pmax) <= max(1.0, 0.001 * max(pmax, 1.0)):
        slice_points = [{"mx_knm": 0.0, "my_knm": 0.0, "theta": 0.0, "angle": 0.0}]

    slice_points = sorted(slice_points, key=lambda point: point["angle"])
    polygon = [(point["mx_knm"], point["my_knm"]) for point in slice_points]
    capacity_radius = _ray_polygon_capacity_radius(polygon, mux_knm, muy_knm)
    demand_radius = math.hypot(mux_knm, muy_knm)
    if demand_radius <= 1e-9:
        pmm_ratio = 0.0
    elif capacity_radius > 0.0 and math.isfinite(capacity_radius):
        pmm_ratio = demand_radius / capacity_radius
    else:
        pmm_ratio = math.inf

    return {
        "ratio": pmm_ratio,
        "capacity_radius_knm": capacity_radius,
        "slice": slice_points,
        "surface": {"mx": mx_grid, "my": my_grid, "p": p_grid},
    }


def analyze_section(
    *,
    width_x_mm: float,
    depth_y_mm: float,
    cover_mm: float,
    bar_dia_mm: float,
    bars_x_face: int,
    bars_y_face: int,
    fc_mpa: float,
    fy_mpa: float,
    es_mpa: float,
    pu_kn: float,
    mux_knm: float,
    muy_knm: float,
    params: CodeParameters,
    biaxial_method: str = "Load contour",
    load_contour_alpha: float = 1.50,
    max_spacing_advisory_mm: float = math.inf,
    sample_count: int = 180,
) -> SectionCheck:
    bars = make_perimeter_bars(
        width_x_mm=width_x_mm,
        depth_y_mm=depth_y_mm,
        cover_mm=cover_mm,
        bar_dia_mm=bar_dia_mm,
        bars_x_face=bars_x_face,
        bars_y_face=bars_y_face,
    )
    as_total = sum(bar.area_mm2 for bar in bars)
    ag = width_x_mm * depth_y_mm
    rho = as_total / ag * 100.0
    clear_x, clear_y, min_clear = clear_spacing_by_face_mm(
        width_x_mm,
        depth_y_mm,
        cover_mm,
        bar_dia_mm,
        bars_x_face,
        bars_y_face,
    )
    spacing_x, spacing_y, max_spacing = center_spacing_by_face_mm(
        width_x_mm,
        depth_y_mm,
        cover_mm,
        bar_dia_mm,
        bars_x_face,
        bars_y_face,
    )
    spacing_status = "OK" if max_spacing <= max_spacing_advisory_mm else "NG"
    pmax = _phi_pmax_kn(width_x_mm, depth_y_mm, bars, fc_mpa, fy_mpa, params)
    curve_x = interaction_curve(
        width_x_mm=width_x_mm,
        depth_y_mm=depth_y_mm,
        bars=bars,
        fc_mpa=fc_mpa,
        fy_mpa=fy_mpa,
        es_mpa=es_mpa,
        params=params,
        axis="x",
        sample_count=sample_count,
    )
    curve_y = interaction_curve(
        width_x_mm=width_x_mm,
        depth_y_mm=depth_y_mm,
        bars=bars,
        fc_mpa=fc_mpa,
        fy_mpa=fy_mpa,
        es_mpa=es_mpa,
        params=params,
        axis="y",
        sample_count=sample_count,
    )
    cap_x = _capacity_at_pu(curve_x, pu_kn)
    cap_y = _capacity_at_pu(curve_y, pu_kn)
    axial_ratio = pu_kn / pmax if pmax > 0 else math.inf
    mux_ratio = abs(mux_knm) / cap_x if cap_x > 0 else math.inf
    muy_ratio = abs(muy_knm) / cap_y if cap_y > 0 else math.inf
    biaxial_linear_ratio = mux_ratio + muy_ratio
    load_contour_ratio = mux_ratio**load_contour_alpha + muy_ratio**load_contour_alpha
    pmm_ratio = math.nan
    pmm_capacity_radius = math.nan
    pmm_surface: dict[str, list[list[float]]] = {"mx": [], "my": [], "p": []}
    pmm_slice: list[dict[str, float]] = []
    if biaxial_method == PMM_METHOD:
        pmm = pmm_surface_check(
            width_x_mm=width_x_mm,
            depth_y_mm=depth_y_mm,
            bars=bars,
            fc_mpa=fc_mpa,
            fy_mpa=fy_mpa,
            es_mpa=es_mpa,
            pu_kn=pu_kn,
            mux_knm=mux_knm,
            muy_knm=muy_knm,
            params=params,
        )
        pmm_ratio = pmm["ratio"]
        pmm_capacity_radius = pmm["capacity_radius_knm"]
        pmm_surface = pmm["surface"]
        pmm_slice = pmm["slice"]
        biaxial_ratio = pmm_ratio
    elif biaxial_method == "Load contour":
        biaxial_ratio = load_contour_ratio
    else:
        biaxial_ratio = biaxial_linear_ratio
    governing_ratio = max(axial_ratio, biaxial_ratio)
    status = "OK" if governing_ratio <= 1.0 else "NG"

    return SectionCheck(
        width_x_mm=width_x_mm,
        depth_y_mm=depth_y_mm,
        as_total_mm2=as_total,
        rho_percent=rho,
        bar_count=len(bars),
        bar_dia_mm=bar_dia_mm,
        fy_mpa=fy_mpa,
        bars_x_face=bars_x_face,
        bars_y_face=bars_y_face,
        clear_spacing_x_mm=clear_x,
        clear_spacing_y_mm=clear_y,
        min_clear_spacing_mm=min_clear,
        center_spacing_x_mm=spacing_x,
        center_spacing_y_mm=spacing_y,
        max_center_spacing_mm=max_spacing,
        max_spacing_advisory_mm=max_spacing_advisory_mm,
        spacing_status=spacing_status,
        phi_pmax_kn=pmax,
        phi_mnx_at_pu_knm=cap_x,
        phi_mny_at_pu_knm=cap_y,
        axial_ratio=axial_ratio,
        mux_ratio=mux_ratio,
        muy_ratio=muy_ratio,
        biaxial_ratio=biaxial_ratio,
        biaxial_linear_ratio=biaxial_linear_ratio,
        load_contour_ratio=load_contour_ratio,
        load_contour_alpha=load_contour_alpha,
        pmm_ratio=pmm_ratio,
        pmm_capacity_radius_knm=pmm_capacity_radius,
        biaxial_method=biaxial_method,
        governing_ratio=governing_ratio,
        status=status,
        curve_x=curve_x,
        curve_y=curve_y,
        pmm_surface=pmm_surface,
        pmm_slice=pmm_slice,
        bars=bars,
    )


def find_reinforcement(
    *,
    width_x_mm: float,
    depth_y_mm: float,
    cover_mm: float,
    bar_dia_options_mm: Iterable[float],
    fc_mpa: float,
    es_mpa: float,
    pu_kn: float,
    mux_knm: float,
    muy_knm: float,
    params: CodeParameters,
    rho_min_percent: float,
    rho_max_percent: float,
    biaxial_method: str = "Load contour",
    load_contour_alpha: float = 1.50,
    min_clear_spacing_mm: float = 75.0,
    max_spacing_advisory_mm: float = math.inf,
    enforce_max_spacing: bool = True,
    max_bars_x_face: int = 28,
    max_bars_y_face: int = 18,
    sample_count: int = 120,
) -> SectionCheck | None:
    candidates: list[tuple[float, float, float, int, int]] = []
    ag = width_x_mm * depth_y_mm
    for dia in bar_dia_options_mm:
        area = bar_area_mm2(dia)
        for nx in range(2, max_bars_x_face + 1):
            for ny in range(2, max_bars_y_face + 1):
                bar_count = 2 * nx + 2 * max(0, ny - 2)
                as_total = bar_count * area
                rho = as_total / ag * 100.0
                _, _, min_clear = clear_spacing_by_face_mm(width_x_mm, depth_y_mm, cover_mm, dia, nx, ny)
                _, _, max_center = center_spacing_by_face_mm(width_x_mm, depth_y_mm, cover_mm, dia, nx, ny)
                if rho_min_percent <= rho <= rho_max_percent:
                    if min_clear >= min_clear_spacing_mm and (not enforce_max_spacing or max_center <= max_spacing_advisory_mm):
                        balance = layout_balance_penalty(width_x_mm, depth_y_mm, cover_mm, dia, nx, ny)
                        candidates.append((as_total, balance, dia, nx, ny))

    for _, _, dia, nx, ny in sorted(candidates, key=lambda item: (item[0], item[1], item[2])):
        try:
            fast_method = "Load contour" if biaxial_method == PMM_METHOD else biaxial_method
            check = analyze_section(
                width_x_mm=width_x_mm,
                depth_y_mm=depth_y_mm,
                cover_mm=cover_mm,
                bar_dia_mm=dia,
                bars_x_face=nx,
                bars_y_face=ny,
                fc_mpa=fc_mpa,
                fy_mpa=rebar_fy_mpa(dia),
                es_mpa=es_mpa,
                pu_kn=pu_kn,
                mux_knm=mux_knm,
                muy_knm=muy_knm,
                params=params,
                biaxial_method=fast_method,
                load_contour_alpha=load_contour_alpha,
                max_spacing_advisory_mm=max_spacing_advisory_mm,
                sample_count=sample_count,
            )
            if biaxial_method == PMM_METHOD:
                if check.axial_ratio > 1.0 or check.mux_ratio > 1.0 or check.muy_ratio > 1.0:
                    continue
                check = analyze_section(
                    width_x_mm=width_x_mm,
                    depth_y_mm=depth_y_mm,
                    cover_mm=cover_mm,
                    bar_dia_mm=dia,
                    bars_x_face=nx,
                    bars_y_face=ny,
                    fc_mpa=fc_mpa,
                    fy_mpa=rebar_fy_mpa(dia),
                    es_mpa=es_mpa,
                    pu_kn=pu_kn,
                    mux_knm=mux_knm,
                    muy_knm=muy_knm,
                    params=params,
                    biaxial_method=biaxial_method,
                    load_contour_alpha=load_contour_alpha,
                    max_spacing_advisory_mm=max_spacing_advisory_mm,
                    sample_count=sample_count,
                )
        except ValueError:
            continue
        if check.status == "OK":
            return check
    return None


COLORS = {
    "concrete": "#d6dee6",
    "concrete_line": "#44515f",
    "pilecap": "#eef2f6",
    "pilecap_line": "#8792a2",
    "bearing": "#13a39a",
    "steel": "#c2410c",
    "axis_x": "#1d4ed8",
    "axis_y": "#be123c",
    "axis_z": "#15803d",
    "grid": "#e5e7eb",
}

DIMENSION_COLOR = "#ff0000"
DIMENSION_TEXT_COLOR = "#00b828"
DIMENSION_LINE_WIDTH = 0.55
DIMENSION_TEXT_SIZE = 7.5
LOAD_TABLE_LINE_GAP_MM = 175.0
FRONT_VIEW_LOAD_TABLE_LINE_GAP_MM = LOAD_TABLE_LINE_GAP_MM * 1.5
SIDE_VIEW_LOAD_TABLE_LINE_GAP_MM = 325.0


def _add_rect(
    fig: go.Figure,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    fillcolor: str,
    linecolor: str,
    dash: str | None = None,
    opacity: float = 1.0,
    layer: str = "below",
) -> None:
    fig.add_shape(
        type="rect",
        layer=layer,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        fillcolor=fillcolor,
        line={"color": linecolor, "width": 2, **({"dash": dash} if dash else {})},
        opacity=opacity,
    )


def _add_axis_arrow(
    fig: go.Figure,
    *,
    x: float,
    y: float,
    dx: float,
    dy: float,
    label: str,
    color: str,
) -> None:
    fig.add_annotation(
        x=x + dx,
        y=y + dy,
        ax=x,
        ay=y,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor=color,
    )
    label_xshift = 14 if abs(dx) >= abs(dy) else 10
    label_yshift = -10 if abs(dx) >= abs(dy) else 14
    fig.add_annotation(
        x=x + dx,
        y=y + dy,
        xref="x",
        yref="y",
        text=label,
        showarrow=False,
        xshift=label_xshift if dx >= 0 else -label_xshift,
        yshift=label_yshift if dy >= 0 else -label_yshift,
        font={"color": color, "size": 12},
    )


def _add_autorange_points(fig: go.Figure, x0: float, x1: float, y0: float, y1: float) -> None:
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="markers",
            marker={"size": 1, "opacity": 0.0},
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _format_mm(value: float, label: str | None = None) -> str:
    return f"{abs(value):.0f}"


def _add_dimension_line(
    fig: go.Figure,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text: str,
    ext0: tuple[float, float] | None = None,
    ext1: tuple[float, float] | None = None,
    color: str = DIMENSION_COLOR,
    text_xshift: int = 0,
    text_yshift: int = 0,
) -> None:
    if math.hypot(x1 - x0, y1 - y0) <= 1e-9:
        return
    line_style = {"color": color, "width": DIMENSION_LINE_WIDTH}
    projection_line_style = {"color": color, "width": DIMENSION_LINE_WIDTH, "dash": "dot"}
    length = math.hypot(x1 - x0, y1 - y0)
    extension_overrun = min(180.0, max(56.0, length * 0.036))
    extension_start_gap = min(32.0, max(10.0, length * 0.004))
    extension_tail = min(1040.0, max(380.0, length * 0.180))
    extension_edge_tick = min(420.0, max(168.0, length * 0.048))
    for source, target in ((ext0, (x0, y0)), (ext1, (x1, y1))):
        if source is None:
            continue
        extension_length = math.hypot(target[0] - source[0], target[1] - source[1])
        if extension_length > 1e-9:
            ux = (target[0] - source[0]) / extension_length
            uy = (target[1] - source[1]) / extension_length
            edge_start = (source[0] + ux * extension_start_gap, source[1] + uy * extension_start_gap)
            edge_end = (
                source[0] + ux * min(extension_start_gap + extension_edge_tick, extension_length),
                source[1] + uy * min(extension_start_gap + extension_edge_tick, extension_length),
            )
            tail_start_distance = min(extension_length, max(extension_start_gap, extension_length - extension_tail))
            tail_start = (source[0] + ux * tail_start_distance, source[1] + uy * tail_start_distance)
            tail_end = (target[0] + ux * extension_overrun, target[1] + uy * extension_overrun)
            segments = [(tail_start, tail_end)]
            if extension_length > extension_tail + extension_edge_tick + extension_start_gap:
                segments.insert(0, (edge_start, edge_end))
        else:
            segments = [(source, target)]
        for segment_start, segment_end in segments:
            fig.add_shape(
                type="line",
                layer="above",
                x0=segment_start[0],
                y0=segment_start[1],
                x1=segment_end[0],
                y1=segment_end[1],
                line=projection_line_style,
            )
    arrow_style = {
        "xref": "x",
        "yref": "y",
        "axref": "x",
        "ayref": "y",
        "text": "",
        "showarrow": True,
        "arrowhead": 3,
        "arrowsize": 1.70,
        "arrowwidth": DIMENSION_LINE_WIDTH,
        "arrowcolor": color,
    }
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, **arrow_style)
    fig.add_annotation(x=x0, y=y0, ax=x1, ay=y1, **arrow_style)
    is_vertical = abs(x1 - x0) < abs(y1 - y0)
    label_xshift = text_xshift
    label_yshift = text_yshift
    if is_vertical and label_xshift == 0:
        label_xshift = -7
    if not is_vertical and label_yshift == 0:
        label_yshift = 6
    fig.add_annotation(
        x=(x0 + x1) / 2.0,
        y=(y0 + y1) / 2.0,
        text=text,
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        xshift=label_xshift,
        yshift=label_yshift,
        textangle=-90 if is_vertical else 0,
        font={"color": DIMENSION_TEXT_COLOR, "size": DIMENSION_TEXT_SIZE},
    )
    xs = [x0, x1]
    ys = [y0, y1]
    if ext0 is not None:
        xs.append(ext0[0])
        ys.append(ext0[1])
    if ext1 is not None:
        xs.append(ext1[0])
        ys.append(ext1[1])
    pad = max(40.0, math.hypot(x1 - x0, y1 - y0) * 0.015)
    _add_autorange_points(fig, min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)


def _unique_sorted_positions(records: Iterable[dict], key: str) -> list[float]:
    return sorted({round(float(row.get(key, 0.0)), 3) for row in records})


def _max_abs_component(records: Iterable[dict], keys: Iterable[str]) -> float:
    values: list[float] = []
    for row in records:
        for key in keys:
            values.append(abs(float(row.get(key, 0.0))))
    return max(values) if values else 0.0


def _scaled_load_delta(value: float, max_abs: float, max_length: float, min_length: float) -> float:
    if abs(value) <= 1e-9 or max_abs <= 1e-9:
        return 0.0
    length = min_length + (max_length - min_length) * min(1.0, abs(value) / max_abs)
    return math.copysign(length, value)


def _load_label(name: str, value: float, unit: str) -> str:
    return f"{name} {value:+.0f} {unit}"


def _compact_load_value(value: float) -> str:
    return f"{value:+.0f}" if abs(value) > 1e-9 else "-"


def _add_load_arrow(
    fig: go.Figure,
    *,
    x: float,
    y: float,
    dx: float,
    dy: float,
    label: str,
    color: str,
    text_xshift: int = 0,
    text_yshift: int = 0,
) -> None:
    if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
        return
    fig.add_annotation(
        x=x + dx,
        y=y + dy,
        ax=x,
        ay=y,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text=label,
        showarrow=True,
        arrowhead=2,
        arrowsize=0.78,
        arrowwidth=1.75,
        arrowcolor=color,
        xshift=text_xshift,
        yshift=text_yshift,
        font={"color": color, "size": 9},
    )


def _add_load_tag(
    fig: go.Figure,
    *,
    x: float,
    y: float,
    text: str,
    color: str,
    xanchor: str = "center",
    yanchor: str = "middle",
    text_xshift: int = 0,
    text_yshift: int = 0,
    boxed: bool = False,
    bordercolor: str | None = None,
    bgcolor: str = "rgba(255,255,255,0.94)",
) -> None:
    if not text:
        return
    annotation = {
        "x": x,
        "y": y,
        "text": text,
        "showarrow": False,
        "xanchor": xanchor,
        "yanchor": yanchor,
        "xshift": text_xshift,
        "yshift": text_yshift,
        "align": "left",
        "font": {"color": color, "size": 9},
    }
    if boxed:
        annotation.update(
            {
                "bgcolor": bgcolor,
                "bordercolor": bordercolor or color,
                "borderwidth": 0.75,
                "borderpad": 5,
            }
        )
    fig.add_annotation(**annotation)


def _add_moment_arc(
    fig: go.Figure,
    *,
    x: float,
    y: float,
    radius: float,
    moment: float,
    label: str,
    color: str,
    text_xshift: int = 0,
    text_yshift: int = 0,
) -> None:
    if abs(moment) <= 1e-9:
        return
    start = 0.16 * math.pi
    stop = 0.84 * math.pi
    if moment < 0.0:
        start, stop = stop, start
    angles = np.linspace(start, stop, 24)
    xs = [x + radius * math.cos(angle) for angle in angles]
    ys = [y + radius * math.sin(angle) for angle in angles]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": color, "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    arrow_text = _load_label(label, moment, "kN-m") if label else ""
    fig.add_annotation(
        x=xs[-1],
        y=ys[-1],
        ax=xs[-2],
        ay=ys[-2],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text=arrow_text,
        showarrow=True,
        arrowhead=2,
        arrowsize=0.8,
        arrowwidth=1.75,
        arrowcolor=color,
        xshift=text_xshift,
        yshift=text_yshift,
        font={"color": color, "size": 9},
    )


def _add_load_legend(fig: go.Figure, text: str) -> None:
    fig.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        text=text,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font={"color": "#334155", "size": 9},
    )


def _finish_view(
    fig: go.Figure,
    title: str,
    x_title: str,
    y_title: str,
    *,
    show_zero_axes: bool = True,
) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        height=480,
        margin={"l": 24, "r": 24, "t": 54, "b": 24},
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        font={"family": "Arial, sans-serif", "size": 13, "color": "#172033"},
    )
    fig.update_xaxes(
        title=x_title,
        showgrid=True,
        gridcolor=COLORS["grid"],
        zeroline=show_zero_axes,
        zerolinecolor="#111827",
        zerolinewidth=1,
    )
    fig.update_yaxes(
        title=y_title,
        showgrid=True,
        gridcolor=COLORS["grid"],
        zeroline=show_zero_axes,
        zerolinecolor="#111827",
        zerolinewidth=1,
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def plan_view(
    bearings: Iterable[dict],
    *,
    width_x_mm: float,
    depth_y_mm: float,
    pilecap_overhang_mm: float,
    bearing_size_x_mm: float,
    bearing_size_y_mm: float,
    selected_bearing_names: Iterable[str] | None = None,
    strip_center_x_mm: float | None = None,
    strip_width_x_mm: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    bearing_records = list(bearings)
    pile_x = width_x_mm / 2.0 + pilecap_overhang_mm
    pile_y = depth_y_mm / 2.0 + pilecap_overhang_mm
    abut_x = width_x_mm / 2.0
    abut_y = depth_y_mm / 2.0

    _add_rect(fig, x0=-pile_x, x1=pile_x, y0=-pile_y, y1=pile_y, fillcolor=COLORS["pilecap"], linecolor=COLORS["pilecap_line"], dash="dash", opacity=0.85)
    _add_rect(fig, x0=-abut_x, x1=abut_x, y0=-abut_y, y1=abut_y, fillcolor=COLORS["concrete"], linecolor=COLORS["concrete_line"])
    if strip_center_x_mm is not None and strip_width_x_mm is not None:
        strip_x0 = max(-abut_x, strip_center_x_mm - strip_width_x_mm / 2.0)
        strip_x1 = min(abut_x, strip_center_x_mm + strip_width_x_mm / 2.0)
        _add_rect(
            fig,
            x0=strip_x0,
            x1=strip_x1,
            y0=-abut_y,
            y1=abut_y,
            fillcolor="#fef3c7",
            linecolor="#d97706",
            dash="dot",
            opacity=0.42,
        )

    half_x = bearing_size_x_mm / 2.0
    half_y = bearing_size_y_mm / 2.0
    bearing_plan_dim = max(bearing_size_x_mm, bearing_size_y_mm)
    selected_names = {str(name) for name in selected_bearing_names or []}
    for row in bearing_records:
        x = float(row.get("x_mm", 0.0))
        y = float(row.get("y_mm", 0.0))
        name = str(row.get("name", "B"))
        is_selected = name in selected_names
        _add_rect(
            fig,
            x0=x - half_x,
            x1=x + half_x,
            y0=y - half_y,
            y1=y + half_y,
            fillcolor="#f59e0b" if is_selected else COLORS["bearing"],
            linecolor="#92400e" if is_selected else "#065f5b",
            opacity=0.98 if is_selected else 0.95,
        )
        fig.add_annotation(x=x, y=y, text=name, showarrow=False, font={"color": "white", "size": 11})

    row_groups: dict[float, list[dict]] = {}
    for row in bearing_records:
        row_key = round(float(row.get("y_mm", 0.0)), 3)
        row_groups.setdefault(row_key, []).append(row)
    grouped_rows = [
        (row_y, sorted(rows, key=lambda item: float(item.get("x_mm", 0.0))))
        for row_y, rows in sorted(row_groups.items(), key=lambda item: item[0], reverse=True)
    ]
    row_count = len(grouped_rows)
    load_value_gap = max(max(half_x, half_y) * 2.45, min(440.0, depth_y_mm * 0.34))
    load_line_gap = LOAD_TABLE_LINE_GAP_MM
    load_components = [
        ("Pu_x", "Pu_x_kN", "kN", COLORS["axis_x"]),
        ("Pu_y", "Pu_y_kN", "kN", COLORS["axis_y"]),
        ("Pu_z", "Pu_z_kN", "kN", COLORS["axis_z"]),
        ("Mu_x", "Mu_x_kNm", "kN-m", "#7c3aed"),
        ("Mu_y", "Mu_y_kNm", "kN-m", "#7c3aed"),
    ]
    load_text_ys: list[float] = []
    load_text_xs: list[float] = []
    active_components = [
        component
        for component in load_components
        if any(abs(float(row.get(component[1], 0.0))) > 1e-9 for _, rows in grouped_rows for row in rows)
    ]
    active_components = active_components or load_components[:3]
    row_block_height = load_line_gap * (len(active_components) + 1)
    group_gap = max(150.0, load_line_gap * 1.25)
    for row_index, (row_y, rows) in enumerate(grouped_rows):
        x_positions = [float(row.get("x_mm", 0.0)) for row in rows]
        if not x_positions:
            continue
        label_x = min(x_positions) - max(360.0, width_x_mm * 0.045)
        row_prefixes = {
            "".join(ch for ch in str(row.get("name", "")) if not ch.isdigit()).strip()
            for row in rows
        }
        row_prefixes.discard("")
        row_title = f"{sorted(row_prefixes)[0]} row" if len(row_prefixes) == 1 else f"y {row_y:+.0f}"
        if row_count > 1:
            header_y = abut_y + load_value_gap + load_line_gap * len(active_components)
            header_y += (row_count - 1 - row_index) * (row_block_height + group_gap)
        else:
            header_y = -abut_y - load_value_gap
        row_ys = [header_y]
        if row_count > 1:
            row_ys += [header_y - load_line_gap * (index + 1) for index in range(len(active_components))]
        else:
            row_ys += [header_y - load_line_gap * (index + 1) for index in range(len(active_components))]
        load_text_ys.extend(row_ys)
        load_text_xs.extend([label_x, *x_positions])

        _add_load_tag(
            fig,
            x=label_x,
            y=header_y,
            text=f"<b>{row_title}</b>",
            color="#334155",
            xanchor="right",
            yanchor="middle",
        )
        for row_item, x in zip(rows, x_positions):
            _add_load_tag(
                fig,
                x=x,
                y=header_y,
                text=f"<b>{row_item.get('name', '')}</b>",
                color="#334155",
                xanchor="center",
                yanchor="middle",
            )

        for component_index, (label, key, unit, color) in enumerate(active_components, start=1):
            component_y = header_y - load_line_gap * component_index
            _add_load_tag(
                fig,
                x=label_x,
                y=component_y,
                text=f"{label} ({unit})",
                color=color,
                xanchor="right",
                yanchor="middle",
            )
            for row_item, x in zip(rows, x_positions):
                _add_load_tag(
                    fig,
                    x=x,
                    y=component_y,
                    text=_compact_load_value(float(row_item.get(key, 0.0))),
                    color=color,
                    xanchor="center",
                    yanchor="middle",
                )

    dim_gap = max(360.0, min(width_x_mm, depth_y_mm) * 0.24, bearing_plan_dim * 2.20)
    x_centers = _unique_sorted_positions(bearing_records, "x_mm")
    y_centers = _unique_sorted_positions(bearing_records, "y_mm")
    right_dim_x = abut_x + dim_gap * (2.60 if len(y_centers) > 1 else 1.65)
    _add_dimension_line(
        fig,
        x0=right_dim_x,
        y0=-abut_y,
        x1=right_dim_x,
        y1=abut_y,
        text=_format_mm(depth_y_mm, "Abutment t"),
        ext0=(abut_x, -abut_y),
        ext1=(abut_x, abut_y),
        text_xshift=7 if len(y_centers) > 1 else -7,
    )

    bearing_dim_y = max(abut_y + dim_gap * 1.10, (max(load_text_ys) if load_text_ys else abut_y) + dim_gap * 0.80)
    if x_centers:
        x_chain = [-abut_x, *x_centers, abut_x]
        for left_x, right_x in zip(x_chain[:-1], x_chain[1:]):
            _add_dimension_line(
                fig,
                x0=left_x,
                y0=bearing_dim_y,
                x1=right_x,
                y1=bearing_dim_y,
                text=_format_mm(right_x - left_x),
                ext0=(left_x, abut_y),
                ext1=(right_x, abut_y),
                text_yshift=4,
            )
    abutment_dim_y = bearing_dim_y + dim_gap * 0.58
    _add_dimension_line(
        fig,
        x0=-abut_x,
        y0=abutment_dim_y,
        x1=abut_x,
        y1=abutment_dim_y,
        text=_format_mm(width_x_mm, "Abutment"),
        ext0=(-abut_x, abut_y),
        ext1=(abut_x, abut_y),
        text_yshift=4,
    )
    if len(y_centers) > 1:
        row_dim_x = abut_x + dim_gap * 1.65
        y_chain = [-abut_y, *y_centers, abut_y]
        for low_y, high_y in zip(y_chain[:-1], y_chain[1:]):
            _add_dimension_line(
                fig,
                x0=row_dim_x,
                y0=low_y,
                x1=row_dim_x,
                y1=high_y,
                text=_format_mm(high_y - low_y),
                ext0=(abut_x, low_y),
                ext1=(abut_x, high_y),
                text_xshift=-7,
            )

    axis_gap = max(850.0, max(width_x_mm, depth_y_mm) * 0.16)
    axis_origin_x = -pile_x - axis_gap
    axis_origin_y = -pile_y - axis_gap
    arrow_x = min(max(width_x_mm * 0.18, 700.0), axis_gap * 0.95)
    arrow_y = min(max(depth_y_mm * 0.55, 420.0), axis_gap * 0.95)
    _add_axis_arrow(fig, x=axis_origin_x, y=axis_origin_y, dx=arrow_x, dy=0, label="+x", color=COLORS["axis_x"])
    _add_axis_arrow(fig, x=axis_origin_x, y=axis_origin_y, dx=0, dy=arrow_y, label="+y", color=COLORS["axis_y"])

    pad = max(width_x_mm, depth_y_mm) * 0.12
    text_y_min = min(load_text_ys) if load_text_ys else -pile_y
    text_y_max = max(load_text_ys) if load_text_ys else pile_y
    _add_autorange_points(
        fig,
        min(axis_origin_x - pad * 0.25, min(load_text_xs) - pad * 0.28 if load_text_xs else -pile_x - pad, -pile_x - pad),
        max(max(load_text_xs) + pad * 0.28 if load_text_xs else pile_x + pad, pile_x + pad),
        min(axis_origin_y - pad * 0.25, text_y_min - pad * 0.18, -pile_y - pad),
        max(text_y_max + pad * 0.18, pile_y + pad),
    )
    return _finish_view(fig, "Section plan at bearing level", "x (mm)", "y (mm)", show_zero_axes=False)


def front_view(
    bearings: Iterable[dict],
    *,
    width_x_mm: float,
    height_z_mm: float,
    pilecap_overhang_mm: float,
    pilecap_thickness_mm: float,
    bearing_size_x_mm: float,
    bearing_size_y_mm: float,
) -> go.Figure:
    fig = go.Figure()
    bearing_records = list(bearings)
    pile_x = width_x_mm / 2.0 + pilecap_overhang_mm
    abut_x = width_x_mm / 2.0
    half_x = bearing_size_x_mm / 2.0
    bearing_display_dim = max(bearing_size_x_mm, bearing_size_y_mm)
    bearing_h = max(80.0, bearing_display_dim * 0.28)

    _add_rect(fig, x0=-pile_x, x1=pile_x, y0=-pilecap_thickness_mm, y1=0, fillcolor=COLORS["pilecap"], linecolor=COLORS["pilecap_line"], dash="dash", opacity=0.85)
    _add_rect(fig, x0=-abut_x, x1=abut_x, y0=0, y1=height_z_mm, fillcolor=COLORS["concrete"], linecolor=COLORS["concrete_line"])

    for row in bearing_records:
        x = float(row.get("x_mm", 0.0))
        z = float(row.get("z_mm", height_z_mm))
        name = str(row.get("name", "B"))
        _add_rect(fig, x0=x - half_x, x1=x + half_x, y0=z, y1=z + bearing_h, fillcolor=COLORS["bearing"], linecolor="#065f5b")
        fig.add_annotation(x=x, y=z + bearing_h / 2.0, text=name, showarrow=False, font={"color": "white", "size": 11})

    front_force_max = _max_abs_component(bearing_records, ("Pu_x_kN", "Pu_z_kN"))
    arrow_max = min(max(max(width_x_mm, height_z_mm) * 0.055, 220.0), 500.0)
    arrow_min = min(110.0, arrow_max * 0.45)
    moment_radius = max(155.0, bearing_display_dim * 0.82)
    load_clearance = max(46.0, half_x * 0.42)
    load_x_extents = [-pile_x, pile_x]
    load_z_extents = [-pilecap_thickness_mm, height_z_mm + bearing_h]

    def representative_load(rows: list[dict], key: str) -> float:
        values = [float(row.get(key, 0.0)) for row in rows]
        resultant = sum(values)
        if abs(resultant) > 1e-9:
            return resultant
        return max(values, key=lambda value: abs(value), default=0.0)

    projected_groups: dict[tuple[float, float], list[dict]] = {}
    for row in bearing_records:
        projected_key = (round(float(row.get("x_mm", 0.0)), 3), round(float(row.get("z_mm", height_z_mm)), 3))
        projected_groups.setdefault(projected_key, []).append(row)

    for (x_key, z_key), rows in sorted(projected_groups.items(), key=lambda item: item[0][0]):
        x = float(x_key)
        z = float(z_key)
        px = representative_load(rows, "Pu_x_kN")
        pz = representative_load(rows, "Pu_z_kN")
        my = representative_load(rows, "Mu_y_kNm")
        load_head_z = z + bearing_h + load_clearance
        px_delta = _scaled_load_delta(px, front_force_max, arrow_max, arrow_min)
        if abs(px_delta) > 1e-9:
            _add_load_arrow(
                fig,
                x=x,
                y=load_head_z,
                dx=px_delta,
                dy=0.0,
                label="",
                color=COLORS["axis_x"],
            )
            load_x_extents.extend([x, x + px_delta])
            load_z_extents.append(load_head_z)
        pz_delta = _scaled_load_delta(pz, front_force_max, arrow_max, arrow_min)
        pz_len = abs(pz_delta)
        if pz_len > 1e-9:
            pz_tail_z = load_head_z + pz_len if pz >= 0.0 else load_head_z
            pz_dz = -pz_len if pz >= 0.0 else pz_len
            _add_load_arrow(
                fig,
                x=x,
                y=pz_tail_z,
                dx=0.0,
                dy=pz_dz,
                label="",
                color=COLORS["axis_z"],
            )
            load_x_extents.append(x)
            load_z_extents.extend([pz_tail_z, pz_tail_z + pz_dz])
        _add_moment_arc(
            fig,
            x=x,
            y=z + bearing_h / 2.0,
            radius=moment_radius,
            moment=my,
            label="",
            color="#7c3aed",
        )
        if abs(my) > 1e-9:
            load_x_extents.extend([x - moment_radius, x + moment_radius])
            load_z_extents.extend([z + bearing_h / 2.0 - moment_radius, z + bearing_h / 2.0 + moment_radius])

    front_load_components = [
        ("Pu_x", "Pu_x_kN", "kN", COLORS["axis_x"]),
        ("Pu_y", "Pu_y_kN", "kN", COLORS["axis_y"]),
        ("Pu_z", "Pu_z_kN", "kN", COLORS["axis_z"]),
        ("Mu_x", "Mu_x_kNm", "kN-m", "#7c3aed"),
        ("Mu_y", "Mu_y_kNm", "kN-m", "#7c3aed"),
    ]
    active_components = [
        component
        for component in front_load_components
        if any(abs(float(row.get(component[1], 0.0))) > 1e-9 for row in bearing_records)
    ]
    active_components = active_components or front_load_components[:3]
    row_groups: dict[float, list[dict]] = {}
    for row in bearing_records:
        row_key = round(float(row.get("y_mm", 0.0)), 3)
        row_groups.setdefault(row_key, []).append(row)
    grouped_rows = [
        (row_y, sorted(rows, key=lambda item: float(item.get("x_mm", 0.0))))
        for row_y, rows in sorted(row_groups.items(), key=lambda item: item[0], reverse=True)
    ]
    load_line_gap = FRONT_VIEW_LOAD_TABLE_LINE_GAP_MM
    row_block_height = load_line_gap * (len(active_components) + 1)
    group_gap = max(220.0, load_line_gap * 1.35)
    table_clearance = max(260.0, load_line_gap * 1.15)
    table_anchor_z = max(load_z_extents) + table_clearance
    load_text_xs: list[float] = []
    load_text_zs: list[float] = []
    for row_index, (row_y, rows) in enumerate(grouped_rows):
        x_positions = [float(row.get("x_mm", 0.0)) for row in rows]
        if not x_positions:
            continue
        label_x = min(x_positions) - max(360.0, width_x_mm * 0.045)
        row_prefixes = {
            "".join(ch for ch in str(row.get("name", "")) if not ch.isdigit()).strip()
            for row in rows
        }
        row_prefixes.discard("")
        row_title = f"{sorted(row_prefixes)[0]} row" if len(row_prefixes) == 1 else f"y {row_y:+.0f}"
        header_z = table_anchor_z + load_line_gap * len(active_components)
        header_z += (len(grouped_rows) - 1 - row_index) * (row_block_height + group_gap)
        load_text_zs.append(header_z)
        load_text_xs.extend([label_x, *x_positions])
        _add_load_tag(
            fig,
            x=label_x,
            y=header_z,
            text=f"<b>{row_title}</b>",
            color="#334155",
            xanchor="right",
            yanchor="middle",
        )
        for row_item, x in zip(rows, x_positions):
            _add_load_tag(
                fig,
                x=x,
                y=header_z,
                text=f"<b>{row_item.get('name', '')}</b>",
                color="#334155",
                xanchor="center",
                yanchor="middle",
            )
        for component_index, (label, key, unit, color) in enumerate(active_components, start=1):
            component_z = header_z - load_line_gap * component_index
            load_text_zs.append(component_z)
            _add_load_tag(
                fig,
                x=label_x,
                y=component_z,
                text=f"{label} ({unit})",
                color=color,
                xanchor="right",
                yanchor="middle",
            )
            for row_item, x in zip(rows, x_positions):
                _add_load_tag(
                    fig,
                    x=x,
                    y=component_z,
                    text=_compact_load_value(float(row_item.get(key, 0.0))),
                    color=color,
                    xanchor="center",
                    yanchor="middle",
                )

    dim_gap = max(360.0, min(width_x_mm, height_z_mm) * 0.12, bearing_display_dim * 2.00)
    right_dim_x = abut_x + dim_gap * 1.65
    _add_dimension_line(
        fig,
        x0=right_dim_x,
        y0=0.0,
        x1=right_dim_x,
        y1=height_z_mm,
        text=_format_mm(height_z_mm, "Height"),
        ext0=(abut_x, 0.0),
        ext1=(abut_x, height_z_mm),
        text_xshift=-7,
    )

    x_centers = _unique_sorted_positions(bearing_records, "x_mm")
    bearing_dim_z = max(max(load_text_zs) if load_text_zs else max(load_z_extents), height_z_mm + bearing_h) + dim_gap * 0.90
    if x_centers:
        x_chain = [-abut_x, *x_centers, abut_x]
        for left_x, right_x in zip(x_chain[:-1], x_chain[1:]):
            _add_dimension_line(
                fig,
                x0=left_x,
                y0=bearing_dim_z,
                x1=right_x,
                y1=bearing_dim_z,
                text=_format_mm(right_x - left_x),
                ext0=(left_x, height_z_mm),
                ext1=(right_x, height_z_mm),
                text_yshift=4,
            )
    abutment_dim_z = bearing_dim_z + dim_gap * 1.15
    _add_dimension_line(
        fig,
        x0=-abut_x,
        y0=abutment_dim_z,
        x1=abut_x,
        y1=abutment_dim_z,
        text=_format_mm(width_x_mm, "Abutment"),
        ext0=(-abut_x, height_z_mm),
        ext1=(abut_x, height_z_mm),
        text_yshift=4,
    )

    axis_gap = max(950.0, max(width_x_mm, height_z_mm) * 0.15)
    axis_origin_x = -pile_x - axis_gap
    axis_origin_z = -pilecap_thickness_mm - axis_gap
    arrow_x = min(max(width_x_mm * 0.18, 800.0), axis_gap * 0.95)
    arrow_z = min(max(height_z_mm * 0.20, 700.0), axis_gap * 0.95)
    _add_axis_arrow(fig, x=axis_origin_x, y=axis_origin_z, dx=arrow_x, dy=0, label="+x", color=COLORS["axis_x"])
    _add_axis_arrow(fig, x=axis_origin_x, y=axis_origin_z, dx=0, dy=arrow_z, label="+z", color=COLORS["axis_z"])

    pad = max(width_x_mm, height_z_mm) * 0.10
    _add_autorange_points(
        fig,
        min(axis_origin_x - pad * 0.25, min(load_x_extents) - pad, min(load_text_xs) - pad * 0.28 if load_text_xs else -pile_x - pad),
        max(max(load_x_extents) + pad, max(load_text_xs) + pad * 0.28 if load_text_xs else pile_x + pad),
        min(axis_origin_z - pad * 0.25, min(load_z_extents) - pad * 0.25),
        max(max(load_z_extents) + pad * 0.45, max(load_text_zs) + pad * 0.18 if load_text_zs else max(load_z_extents) + pad),
    )
    return _finish_view(fig, "Front view", "x (mm)", "z (mm)", show_zero_axes=False)


def _add_side_earth_pressure_diagram(
    fig: go.Figure,
    *,
    earth_pressure: EarthPressureSummary,
    abut_y: float,
    height_z_mm: float,
    pile_y: float,
    load_y_extents: list[float],
    load_z_extents: list[float],
) -> None:
    if earth_pressure.height_m <= 1e-9:
        return
    service_soil = abs(earth_pressure.service_soil_kn_per_m)
    service_other = abs(earth_pressure.service_other_kn_per_m)
    service_live = abs(earth_pressure.service_live_kn_per_m)
    if service_soil + service_other + service_live <= 1e-9:
        return

    h_draw = min(max(earth_pressure.height_m * 1000.0, 1.0), max(height_z_mm, 1.0))
    force_sign = 1.0 if earth_pressure.direction == "+y" else -1.0
    soil_side = -force_sign
    face_y = soil_side * abut_y
    max_len = min(max(height_z_mm * 0.18, 360.0), max(620.0, pile_y * 0.58))
    label_gap = max(300.0, max_len * 0.32)

    p_soil_bottom = 2.0 * service_soil / earth_pressure.height_m
    p_other = service_other / earth_pressure.height_m
    p_live = service_live / earth_pressure.height_m
    p_total = max(p_soil_bottom + p_other + p_live, 1e-9)
    len_other = max_len * p_other / p_total
    len_live = max_len * p_live / p_total
    len_soil = max_len * p_soil_bottom / p_total
    uniform_len = len_other + len_live

    def y_at(offset_mm: float) -> float:
        return face_y + soil_side * offset_mm

    if len_other > 1.0:
        _add_rect(
            fig,
            x0=min(y_at(0.0), y_at(len_other)),
            x1=max(y_at(0.0), y_at(len_other)),
            y0=0.0,
            y1=h_draw,
            fillcolor="#bfdbfe",
            linecolor="#2563eb",
            opacity=0.26,
            layer="above",
        )
    if len_live > 1.0:
        _add_rect(
            fig,
            x0=min(y_at(len_other), y_at(uniform_len)),
            x1=max(y_at(len_other), y_at(uniform_len)),
            y0=0.0,
            y1=h_draw,
            fillcolor="#fed7aa",
            linecolor="#f97316",
            opacity=0.28,
            layer="above",
        )
    if len_soil > 1.0:
        fig.add_trace(
            go.Scatter(
                x=[y_at(uniform_len), y_at(uniform_len), y_at(uniform_len + len_soil), y_at(uniform_len)],
                y=[h_draw, 0.0, 0.0, h_draw],
                mode="lines",
                fill="toself",
                fillcolor="rgba(15, 118, 110, 0.24)",
                line={"color": "#0f766e", "width": 1.5},
                hovertemplate=(
                    "EH soil pressure<br>"
                    f"K={earth_pressure.pressure_coefficient:.3f}<br>"
                    f"p_base={p_soil_bottom:.1f} kPa<extra></extra>"
                ),
                showlegend=False,
            )
        )

    for z_frac in (0.18, 0.38, 0.58, 0.78):
        z = h_draw * z_frac
        pressure_len = uniform_len + len_soil * (1.0 - z_frac)
        if pressure_len <= 45.0:
            continue
        _add_load_arrow(
            fig,
            x=y_at(pressure_len),
            y=z,
            dx=-soil_side * pressure_len,
            dy=0.0,
            label="",
            color="#0f766e",
        )

    soil_uls = abs(earth_pressure.uls_soil_vy_kn)
    uniform_uls = abs(earth_pressure.uls_other_vy_kn) + abs(earth_pressure.uls_live_vy_kn)
    total_uls = soil_uls + uniform_uls
    resultant_z = 0.0
    if total_uls > 1e-9:
        resultant_z = (soil_uls * h_draw / 3.0 + uniform_uls * h_draw / 2.0) / total_uls
        resultant_len = max(uniform_len + len_soil * (1.0 - resultant_z / h_draw), max_len * 0.32)
        resultant_label_y = y_at(max_len + label_gap + 1050.0)
        fig.add_shape(
            type="line",
            layer="above",
            x0=face_y,
            y0=resultant_z,
            x1=resultant_label_y,
            y1=resultant_z,
            line={"color": "#b45309", "width": 1.2, "dash": "dot"},
        )
        _add_load_arrow(
            fig,
            x=y_at(resultant_len + max_len * 0.16),
            y=resultant_z,
            dx=-soil_side * (resultant_len + max_len * 0.16),
            dy=0.0,
            label="",
            color="#b45309",
        )
        _add_load_tag(
            fig,
            x=resultant_label_y,
            y=resultant_z,
            text=(
                "<b>ULS resultant</b><br>"
                f"Vy = {earth_pressure.uls_vy_kn:+.0f} kN<br>"
                f"z_R = {resultant_z / 1000.0:.2f} m above base<br>"
                f"Mux = {earth_pressure.uls_mux_knm:+.0f} kN-m"
            ),
            color="#92400e",
            xanchor="right" if soil_side < 0.0 else "left",
            yanchor="middle",
            boxed=True,
            bordercolor="#92400e",
        )

    service_total = service_soil + service_other + service_live
    component_lines = [
        f"EH triangular: {service_soil:.1f} kN/m @ H/3",
        f"p_base = {p_soil_bottom:.1f} kPa",
    ]
    if service_other > 1e-9:
        component_lines.append(f"q surcharge: {service_other:.1f} kN/m @ H/2")
    if service_live > 1e-9:
        component_lines.append(f"LS surcharge: {service_live:.1f} kN/m @ H/2")
    summary_label_y = y_at(max_len + label_gap + 5600.0)
    summary_label_z = resultant_z if resultant_z > 1e-9 else h_draw / 3.0
    _add_load_tag(
        fig,
        x=summary_label_y,
        y=summary_label_z,
        text=(
            "<b>Lateral earth pressure</b><br>"
            f"{earth_pressure.combination}<br>"
            f"K = {earth_pressure.pressure_coefficient:.3f}, H = {earth_pressure.height_m:.2f} m<br>"
            f"Service resultant = {service_total:.1f} kN/m<br>"
            + "<br>".join(component_lines)
        ),
        color="#0f172a",
        xanchor="right" if soil_side < 0.0 else "left",
        yanchor="middle",
        boxed=True,
        bordercolor="#334155",
    )

    outer_y = y_at(max_len + label_gap + 6100.0)
    load_y_extents.extend([face_y, y_at(max_len), outer_y])
    load_z_extents.extend([0.0, h_draw, h_draw + max(260.0, height_z_mm * 0.07), resultant_z, summary_label_z])


def _add_side_approach_slab_reaction(
    fig: go.Figure,
    *,
    approach_slab: ApproachSlabReactionSummary,
    abut_y: float,
    height_z_mm: float,
    bearing_h: float,
    load_y_extents: list[float],
    load_z_extents: list[float],
) -> None:
    soil_side = 1.0 if approach_slab.centroid_y_mm > 0.0 else -1.0
    soil_face_y = soil_side * abut_y
    seat_mm = min(max(approach_slab.seat_length_m * 1000.0, 0.0), max(2.0 * abut_y, 0.0))
    seat_inner_y = soil_face_y - soil_side * seat_mm
    gap_mm = max(approach_slab.gap_m * 1000.0, 0.0)
    length_draw_mm = max(approach_slab.length_y_m * 1000.0 / 3.0, gap_mm, 900.0)
    length_draw_limit = max(8000.0, height_z_mm * 1.60)
    length_draw = min(length_draw_mm, length_draw_limit)
    slab_end_y = soil_face_y + soil_side * length_draw
    gap_end_y = soil_face_y + soil_side * min(gap_mm, length_draw)
    slab_z0 = height_z_mm
    slab_thk = min(max(approach_slab.thickness_m * 1000.0, 75.0), 230.0)
    slab_z1 = slab_z0 + slab_thk

    _add_rect(
        fig,
        x0=min(seat_inner_y, slab_end_y),
        x1=max(seat_inner_y, slab_end_y),
        y0=slab_z0,
        y1=slab_z1,
        fillcolor="#fde68a",
        linecolor="#92400e",
        opacity=0.88,
        layer="above",
    )
    fig.add_shape(
        type="line",
        layer="above",
        x0=soil_face_y,
        y0=slab_z0,
        x1=soil_face_y,
        y1=slab_z1,
        line={"color": "#92400e", "width": 2.0},
    )
    fig.add_shape(
        type="line",
        layer="above",
        x0=gap_end_y,
        y0=slab_z0,
        x1=gap_end_y,
        y1=slab_z1,
        line={"color": "#64748b", "width": 1.4, "dash": "dash"},
    )

    dim_z = slab_z1 + max(90.0, height_z_mm * 0.018)
    fig.add_shape(
        type="line",
        layer="above",
        x0=soil_face_y,
        y0=dim_z,
        x1=gap_end_y,
        y1=dim_z,
        line={"color": "#475569", "width": 1.1, "dash": "dot"},
    )
    gap_note = " shown clipped" if gap_mm > length_draw + 1.0 else ""
    note_y = dim_z + max(80.0, height_z_mm * 0.015)
    gap_label_x = (soil_face_y + gap_end_y) / 2.0
    live_model = approach_slab.live_load_preset.split(" (", 1)[0]
    if live_model.startswith("Road - AASHTO HL-93"):
        live_model = "Road - AASHTO HL-93"
    if approach_slab.live_q_kpa > 1e-9:
        live_text = f"{live_model}: q_LL = {approach_slab.live_q_kpa:.2f} kPa"
    elif approach_slab.live_line_load_kn_m > 1e-9:
        live_text = f"{live_model}: q_LL = {approach_slab.live_line_load_kn_m:.2f} kN/m"
    else:
        live_text = f"{live_model}: q_LL = 0.00"
    _add_load_tag(
        fig,
        x=gap_label_x + soil_side * max(930.0, length_draw * 0.28),
        y=note_y,
        text=live_text,
        color="#475569",
        xanchor="right" if soil_side < 0.0 else "left",
        yanchor="bottom",
    )
    _add_load_tag(
        fig,
        x=gap_label_x,
        y=note_y,
        text=f"l_gap = {approach_slab.gap_m:.2f} m{gap_note}",
        color="#475569",
        xanchor="center",
        yanchor="bottom",
    )

    arrow_len = max(520.0, min(980.0, height_z_mm * 0.18))
    arrow_tail_z = slab_z1 + arrow_len
    if approach_slab.uls_pu_z_kn > 1e-9:
        _add_load_arrow(
            fig,
            x=approach_slab.centroid_y_mm,
            y=arrow_tail_z,
            dx=0.0,
            dy=-(arrow_tail_z - slab_z0),
            label="",
            color="#dc2626",
        )
    else:
        fig.add_shape(
            type="line",
            layer="above",
            x0=approach_slab.centroid_y_mm,
            y0=slab_z0,
            x1=approach_slab.centroid_y_mm,
            y1=arrow_tail_z,
            line={"color": "#dc2626", "width": 1.2, "dash": "dot"},
        )
    label_y = slab_end_y + soil_side * max(260.0, length_draw * 0.08)
    label_z = slab_z0 - max(540.0, height_z_mm * 0.12)
    _add_load_tag(
        fig,
        x=label_y,
        y=label_z,
        text=(
            "<b>Approach slab reaction</b><br>"
            f"Service DC+DW = {approach_slab.service_dc_kn + approach_slab.service_dw_kn:.0f} kN<br>"
            f"Service LL = {approach_slab.service_ll_kn:.0f} kN<br>"
            f"ULS Pu_z = {approach_slab.uls_pu_z_kn:.0f} kN<br>"
            f"y_AS = {approach_slab.centroid_y_mm:+.0f} mm<br>"
            f"Mux = {approach_slab.uls_mux_knm:+.0f} kN-m"
        ),
        color="#991b1b",
        xanchor="right" if soil_side < 0.0 else "left",
        yanchor="middle",
        boxed=True,
        bordercolor="#991b1b",
    )

    load_y_extents.extend(
        [
            seat_inner_y,
            soil_face_y,
            slab_end_y,
            gap_end_y,
            gap_label_x + soil_side * max(930.0, length_draw * 0.28),
            label_y,
            label_y + soil_side * max(1400.0, length_draw * 0.20),
            approach_slab.centroid_y_mm,
        ]
    )
    load_z_extents.extend([label_z, slab_z0, slab_z1, dim_z, arrow_tail_z])


def side_view(
    bearings: Iterable[dict],
    *,
    depth_y_mm: float,
    height_z_mm: float,
    pilecap_overhang_mm: float,
    pilecap_thickness_mm: float,
    bearing_size_x_mm: float,
    bearing_size_y_mm: float,
    earth_pressure: EarthPressureSummary | None = None,
    approach_slab: ApproachSlabReactionSummary | None = None,
) -> go.Figure:
    fig = go.Figure()
    bearing_records = list(bearings)
    pile_y = depth_y_mm / 2.0 + pilecap_overhang_mm
    abut_y = depth_y_mm / 2.0
    half_y = bearing_size_y_mm / 2.0
    bearing_display_dim = max(bearing_size_x_mm, bearing_size_y_mm)
    bearing_h = max(80.0, bearing_display_dim * 0.28)

    _add_rect(fig, x0=-pile_y, x1=pile_y, y0=-pilecap_thickness_mm, y1=0, fillcolor=COLORS["pilecap"], linecolor=COLORS["pilecap_line"], dash="dash", opacity=0.85)
    _add_rect(fig, x0=-abut_y, x1=abut_y, y0=0, y1=height_z_mm, fillcolor=COLORS["concrete"], linecolor=COLORS["concrete_line"])

    for row in bearing_records:
        y = float(row.get("y_mm", 0.0))
        z = float(row.get("z_mm", height_z_mm))
        name = str(row.get("name", "B"))
        _add_rect(fig, x0=y - half_y, x1=y + half_y, y0=z, y1=z + bearing_h, fillcolor=COLORS["bearing"], linecolor="#065f5b")
        fig.add_annotation(x=y, y=z + bearing_h / 2.0, text=name, showarrow=False, font={"color": "white", "size": 11})

    side_force_max = _max_abs_component(bearing_records, ("Pu_y_kN", "Pu_z_kN"))
    base_arrow_max = min(max(max(depth_y_mm, height_z_mm) * 0.060, 210.0), 460.0)
    arrow_max = base_arrow_max * 4.0
    arrow_min = min(105.0, base_arrow_max * 0.45) * 4.0
    moment_radius = max(310.0, bearing_display_dim * 1.64)
    load_clearance = max(46.0, half_y * 0.42)
    load_y_extents = [-pile_y, pile_y]
    load_z_extents = [-pilecap_thickness_mm, height_z_mm + bearing_h]
    if earth_pressure is not None:
        _add_side_earth_pressure_diagram(
            fig,
            earth_pressure=earth_pressure,
            abut_y=abut_y,
            height_z_mm=height_z_mm,
            pile_y=pile_y,
            load_y_extents=load_y_extents,
            load_z_extents=load_z_extents,
        )
    if approach_slab is not None:
        _add_side_approach_slab_reaction(
            fig,
            approach_slab=approach_slab,
            abut_y=abut_y,
            height_z_mm=height_z_mm,
            bearing_h=bearing_h,
            load_y_extents=load_y_extents,
            load_z_extents=load_z_extents,
        )

    def representative_load(rows: list[dict], key: str) -> float:
        values = [float(row.get(key, 0.0)) for row in rows]
        resultant = sum(values)
        if abs(resultant) > 1e-9:
            return resultant
        return max(values, key=lambda value: abs(value), default=0.0)

    projected_groups: dict[tuple[float, float], list[dict]] = {}
    for row in bearing_records:
        projected_key = (round(float(row.get("y_mm", 0.0)), 3), round(float(row.get("z_mm", height_z_mm)), 3))
        projected_groups.setdefault(projected_key, []).append(row)

    for (y_key, z_key), rows in sorted(projected_groups.items(), key=lambda item: item[0][0]):
        y = float(y_key)
        z = float(z_key)
        py = representative_load(rows, "Pu_y_kN")
        pz = representative_load(rows, "Pu_z_kN")
        mx = representative_load(rows, "Mu_x_kNm")
        load_head_z = z + bearing_h + load_clearance
        py_delta = _scaled_load_delta(py, side_force_max, arrow_max, arrow_min)
        if abs(py_delta) > 1e-9:
            _add_load_arrow(
                fig,
                x=y,
                y=load_head_z,
                dx=py_delta,
                dy=0.0,
                label="",
                color=COLORS["axis_y"],
            )
            load_y_extents.extend([y, y + py_delta])
            load_z_extents.append(load_head_z)
        pz_delta = _scaled_load_delta(pz, side_force_max, arrow_max, arrow_min)
        pz_len = abs(pz_delta)
        if pz_len > 1e-9:
            pz_tail_z = load_head_z + pz_len if pz >= 0.0 else load_head_z
            pz_dz = -pz_len if pz >= 0.0 else pz_len
            _add_load_arrow(
                fig,
                x=y,
                y=pz_tail_z,
                dx=0.0,
                dy=pz_dz,
                label="",
                color=COLORS["axis_z"],
            )
            load_y_extents.append(y)
            load_z_extents.extend([pz_tail_z, pz_tail_z + pz_dz])
        _add_moment_arc(
            fig,
            x=y,
            y=z + bearing_h / 2.0,
            radius=moment_radius,
            moment=mx,
            label="",
            color="#7c3aed",
        )
        if abs(mx) > 1e-9:
            load_y_extents.extend([y - moment_radius, y + moment_radius])
            load_z_extents.extend([z + bearing_h / 2.0 - moment_radius, z + bearing_h / 2.0 + moment_radius])

    side_load_components = [
        ("Pu_x", "Pu_x_kN", "kN", COLORS["axis_x"]),
        ("Pu_y", "Pu_y_kN", "kN", COLORS["axis_y"]),
        ("Pu_z", "Pu_z_kN", "kN", COLORS["axis_z"]),
        ("Mu_x", "Mu_x_kNm", "kN-m", "#7c3aed"),
        ("Mu_y", "Mu_y_kNm", "kN-m", "#7c3aed"),
    ]
    active_components = [
        component
        for component in side_load_components
        if any(abs(float(row.get(component[1], 0.0))) > 1e-9 for row in bearing_records)
    ]
    active_components = active_components or side_load_components[:3]
    row_groups: dict[float, list[dict]] = {}
    for row in bearing_records:
        row_key = round(float(row.get("y_mm", 0.0)), 3)
        row_groups.setdefault(row_key, []).append(row)
    grouped_rows = [
        (row_y, sorted(rows, key=lambda item: float(item.get("x_mm", 0.0))))
        for row_y, rows in sorted(row_groups.items(), key=lambda item: item[0], reverse=True)
    ]
    load_line_gap = SIDE_VIEW_LOAD_TABLE_LINE_GAP_MM
    row_block_height = load_line_gap * (len(active_components) + 1)
    group_gap = max(220.0, load_line_gap * 1.35)
    table_clearance = max(500.0, load_line_gap * 1.45)
    table_anchor_z = max(load_z_extents) + table_clearance
    load_text_axis_xs: list[float] = []
    load_text_zs: list[float] = []
    for row_index, (row_y, rows) in enumerate(grouped_rows):
        if not rows:
            continue
        column_gap = max(2200.0, SIDE_VIEW_LOAD_TABLE_LINE_GAP_MM * 3.4, bearing_display_dim * 8.8)
        table_span = column_gap * max(len(rows) - 1, 1)
        table_x_positions = [
            -table_span / 2.0 + index * column_gap
            for index in range(len(rows))
        ]
        label_x = min(table_x_positions) - max(360.0, column_gap * 0.72)
        row_prefixes = {
            "".join(ch for ch in str(row.get("name", "")) if not ch.isdigit()).strip()
            for row in rows
        }
        row_prefixes.discard("")
        row_title = f"{sorted(row_prefixes)[0]} row" if len(row_prefixes) == 1 else f"y {row_y:+.0f}"
        header_z = table_anchor_z + load_line_gap * len(active_components)
        header_z += (len(grouped_rows) - 1 - row_index) * (row_block_height + group_gap)
        load_text_zs.append(header_z)
        load_text_axis_xs.extend([label_x, *table_x_positions])
        _add_load_tag(
            fig,
            x=label_x,
            y=header_z,
            text=f"<b>{row_title}</b>",
            color="#334155",
            xanchor="right",
            yanchor="middle",
        )
        for row_item, x in zip(rows, table_x_positions):
            _add_load_tag(
                fig,
                x=x,
                y=header_z,
                text=f"<b>{row_item.get('name', '')}</b>",
                color="#334155",
                xanchor="center",
                yanchor="middle",
            )
        for component_index, (label, key, unit, color) in enumerate(active_components, start=1):
            component_z = header_z - load_line_gap * component_index
            load_text_zs.append(component_z)
            _add_load_tag(
                fig,
                x=label_x,
                y=component_z,
                text=f"{label} ({unit})",
                color=color,
                xanchor="right",
                yanchor="middle",
            )
            for row_item, x in zip(rows, table_x_positions):
                _add_load_tag(
                    fig,
                    x=x,
                    y=component_z,
                    text=_compact_load_value(float(row_item.get(key, 0.0))),
                    color=color,
                    xanchor="center",
                    yanchor="middle",
                )

    dim_gap = max(360.0, min(depth_y_mm, height_z_mm) * 0.20, bearing_display_dim * 2.00)
    right_dim_y = max(abut_y + dim_gap, max(load_y_extents) + dim_gap * 0.25)
    _add_dimension_line(
        fig,
        x0=right_dim_y,
        y0=0.0,
        x1=right_dim_y,
        y1=height_z_mm,
        text=_format_mm(height_z_mm, "Height"),
        ext0=(abut_y, 0.0),
        ext1=(abut_y, height_z_mm),
        text_xshift=-7,
    )

    y_centers = _unique_sorted_positions(bearing_records, "y_mm")
    bearing_dim_z = max(max(load_text_zs) if load_text_zs else max(load_z_extents), height_z_mm + bearing_h) + dim_gap * 0.90
    if y_centers:
        y_chain = [-abut_y, *y_centers, abut_y]
        for left_y, right_y in zip(y_chain[:-1], y_chain[1:]):
            _add_dimension_line(
                fig,
                x0=left_y,
                y0=bearing_dim_z,
                x1=right_y,
                y1=bearing_dim_z,
                text=_format_mm(right_y - left_y),
                ext0=(left_y, height_z_mm),
                ext1=(right_y, height_z_mm),
                text_yshift=4,
            )
    abutment_dim_z = bearing_dim_z + dim_gap * 1.15
    _add_dimension_line(
        fig,
        x0=-abut_y,
        y0=abutment_dim_z,
        x1=abut_y,
        y1=abutment_dim_z,
        text=_format_mm(depth_y_mm, "Abutment t"),
        ext0=(-abut_y, height_z_mm),
        ext1=(abut_y, height_z_mm),
        text_yshift=4,
    )

    axis_gap = max(950.0, max(depth_y_mm, height_z_mm) * 0.15)
    axis_origin_y = -pile_y - axis_gap
    axis_origin_z = -pilecap_thickness_mm - axis_gap
    arrow_y = min(max(depth_y_mm * 0.55, 500.0), axis_gap * 0.95)
    arrow_z = min(max(height_z_mm * 0.20, 700.0), axis_gap * 0.95)
    _add_axis_arrow(fig, x=axis_origin_y, y=axis_origin_z, dx=arrow_y, dy=0, label="+y", color=COLORS["axis_y"])
    _add_axis_arrow(fig, x=axis_origin_y, y=axis_origin_z, dx=0, dy=arrow_z, label="+z", color=COLORS["axis_z"])

    pad = max(depth_y_mm, height_z_mm) * 0.10
    _add_autorange_points(
        fig,
        min(axis_origin_y - pad * 0.25, min(load_y_extents) - pad, min(load_text_axis_xs) - pad * 0.28 if load_text_axis_xs else -pile_y - pad),
        max(max(load_y_extents) + pad, max(load_text_axis_xs) + pad * 0.28 if load_text_axis_xs else pile_y + pad),
        min(axis_origin_z - pad * 0.25, min(load_z_extents) - pad * 0.25),
        max(max(load_z_extents) + pad * 0.45, max(load_text_zs) + pad * 0.18 if load_text_zs else max(load_z_extents) + pad),
    )
    fig = _finish_view(fig, "Side view", "y (mm)", "z (mm)", show_zero_axes=False)
    fig.update_layout(height=720)
    return fig


def reinforcement_plan(check: SectionCheck) -> go.Figure:
    fig = go.Figure()
    x = check.width_x_mm / 2.0
    y = check.depth_y_mm / 2.0
    _add_rect(fig, x0=-x, x1=x, y0=-y, y1=y, fillcolor="#f8fafc", linecolor=COLORS["concrete_line"])

    display_offset = min(
        max(2.0 * (check.bar_dia_mm + min(check.clear_spacing_x_mm, check.clear_spacing_y_mm) * 0.05), 90.0),
        max(90.0, min(check.width_x_mm, check.depth_y_mm) * 0.22),
    )
    if check.width_x_mm > 2.0 * display_offset and check.depth_y_mm > 2.0 * display_offset:
        sx = (check.width_x_mm - 2.0 * display_offset) / check.width_x_mm
        sy = (check.depth_y_mm - 2.0 * display_offset) / check.depth_y_mm
    else:
        sx = sy = 1.0

    top_bottom_bars: list[Bar] = []
    side_bars: list[Bar] = []
    corner_tol = 1e-6
    for bar in check.bars:
        plotted = Bar(x_mm=bar.x_mm * sx, y_mm=bar.y_mm * sy, area_mm2=bar.area_mm2)
        is_top_bottom = abs(abs(bar.y_mm) - max(abs(b.y_mm) for b in check.bars)) <= corner_tol
        if is_top_bottom:
            top_bottom_bars.append(plotted)
        else:
            side_bars.append(plotted)

    marker_size = max(2.0, min(3.4, check.bar_dia_mm * 0.09))
    common_hover = "DB%{customdata[0]:.0f}<br>x=%{customdata[1]:.0f} mm<br>y=%{customdata[2]:.0f} mm<extra></extra>"
    fig.add_trace(
        go.Scatter(
            x=[bar.x_mm for bar in top_bottom_bars],
            y=[bar.y_mm for bar in top_bottom_bars],
            mode="markers",
            name="top/bottom bars",
            marker={
                "size": marker_size,
                "color": "#2563eb",
                "line": {"color": "#1e3a8a", "width": 1.0},
                "opacity": 1.0,
            },
            hovertemplate=common_hover,
            customdata=[
                [check.bar_dia_mm, original.x_mm, original.y_mm]
                for original in check.bars
                if abs(abs(original.y_mm) - max(abs(b.y_mm) for b in check.bars)) <= corner_tol
            ],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[bar.x_mm for bar in side_bars],
            y=[bar.y_mm for bar in side_bars],
            mode="markers",
            name="left/right bars",
            marker={
                "size": marker_size,
                "color": "#dc2626",
                "line": {"color": "#7f1d1d", "width": 1.0},
                "opacity": 1.0,
            },
            hovertemplate=common_hover,
            customdata=[
                [check.bar_dia_mm, original.x_mm, original.y_mm]
                for original in check.bars
                if abs(abs(original.y_mm) - max(abs(b.y_mm) for b in check.bars)) > corner_tol
            ],
        )
    )
    spacing_status_color = "#0f766e" if check.spacing_status == "OK" else "#b91c1c"
    max_spacing_text = (
        f"{check.max_spacing_advisory_mm:.0f} mm"
        if math.isfinite(check.max_spacing_advisory_mm)
        else "not set"
    )
    label = (
        f"Top/bottom faces: {check.bars_x_face} {rebar_label(check.bar_dia_mm)} each<br>"
        f"Left/right faces: {check.bars_y_face} {rebar_label(check.bar_dia_mm)} each<br>"
        f"Total: {check.bar_count} bars, fy = {check.fy_mpa:.0f} MPa, rho = {check.rho_percent:.3f}%<br>"
        f"Clear spacing: x-face {check.clear_spacing_x_mm:.0f} mm, y-face {check.clear_spacing_y_mm:.0f} mm<br>"
        f"c/c spacing: x-face {check.center_spacing_x_mm:.0f} mm, y-face {check.center_spacing_y_mm:.0f} mm<br>"
        f"Max spacing advisory: {max_spacing_text} "
        f"<span style='color:{spacing_status_color}'>{check.spacing_status}</span>"
    )
    fig.add_annotation(
        x=-x,
        y=y + max(check.depth_y_mm, 450.0) * 0.28,
        text=label,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        align="left",
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#cbd5e1",
        borderwidth=1,
        borderpad=6,
        font={"color": "#172033", "size": 13},
    )
    axis_gap = max(360.0, check.depth_y_mm * 0.38)
    axis_origin_x = -x
    axis_origin_y = -y - axis_gap
    arrow_x = min(max(check.width_x_mm * 0.14, 500.0), 1300.0)
    arrow_y = min(max(check.depth_y_mm * 0.25, 250.0), axis_gap * 0.72)
    _add_axis_arrow(fig, x=axis_origin_x, y=axis_origin_y, dx=arrow_x, dy=0, label="+x", color=COLORS["axis_x"])
    _add_axis_arrow(fig, x=axis_origin_x, y=axis_origin_y, dx=0, dy=arrow_y, label="+y", color=COLORS["axis_y"])
    pad = max(check.width_x_mm, check.depth_y_mm) * 0.12
    fig.update_xaxes(range=[-x - pad, x + pad])
    fig.update_yaxes(range=[-y - axis_gap - 220.0, y + pad * 1.65])
    fig = _finish_view(fig, "Base section reinforcement", "x (mm)", "y (mm)")
    fig.update_layout(showlegend=True, legend={"orientation": "h", "y": 1.02, "x": 0.52, "xanchor": "center"})
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)
    return fig


def interaction_plot(check: SectionCheck, pu_kn: float, mux_knm: float, muy_knm: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p["phi_mn_knm"] for p in check.curve_x], y=[p["phi_pn_kn"] for p in check.curve_x], mode="lines", name="about x", line={"color": COLORS["axis_x"], "width": 3}))
    fig.add_trace(go.Scatter(x=[p["phi_mn_knm"] for p in check.curve_y], y=[p["phi_pn_kn"] for p in check.curve_y], mode="lines", name="about y", line={"color": COLORS["axis_y"], "width": 3}))
    fig.add_trace(go.Scatter(x=[abs(mux_knm)], y=[pu_kn], mode="markers", name="Pu, Mux", marker={"size": 12, "color": COLORS["axis_x"], "symbol": "x"}))
    fig.add_trace(go.Scatter(x=[abs(muy_knm)], y=[pu_kn], mode="markers", name="Pu, Muy", marker={"size": 12, "color": COLORS["axis_y"], "symbol": "x"}))
    fig.update_layout(
        title={"text": "Uniaxial interaction curves at base section", "x": 0.02, "xanchor": "left"},
        height=440,
        margin={"l": 24, "r": 24, "t": 54, "b": 24},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={"orientation": "h", "y": 1.02, "x": 0.52, "xanchor": "center"},
        font={"family": "Arial, sans-serif", "size": 13, "color": "#172033"},
    )
    fig.update_xaxes(title="phi Mn (kN-m)", gridcolor=COLORS["grid"], zeroline=True)
    fig.update_yaxes(title="phi Pn (kN)", gridcolor=COLORS["grid"], zeroline=True)
    return fig


def biaxial_load_contour_plot(check: SectionCheck, mux_knm: float, muy_knm: float) -> go.Figure:
    cap_x = max(check.phi_mnx_at_pu_knm, 1e-9)
    cap_y = max(check.phi_mny_at_pu_knm, 1e-9)
    alpha = check.load_contour_alpha
    theta_values = np.linspace(0.0, 2.0 * math.pi, 241)
    contour_x = []
    contour_y = []
    for theta in theta_values:
        c = math.cos(theta)
        s = math.sin(theta)
        denom = (abs(c) / cap_x) ** alpha + (abs(s) / cap_y) ** alpha
        radius = denom ** (-1.0 / alpha)
        contour_x.append(radius * c)
        contour_y.append(radius * s)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=contour_x,
            y=contour_y,
            mode="lines",
            fill="toself",
            fillcolor="rgba(15, 118, 110, 0.08)",
            name=f"load contour alpha={alpha:.2f}",
            line={"color": "#0f766e", "width": 3},
        )
    )
    if alpha > 1.01:
        diamond_x = [cap_x, 0.0, -cap_x, 0.0, cap_x]
        diamond_y = [0.0, cap_y, 0.0, -cap_y, 0.0]
        fig.add_trace(
            go.Scatter(
                x=diamond_x,
                y=diamond_y,
                mode="lines",
                name="linear contour alpha=1.00",
                line={"color": "#94a3b8", "width": 2, "dash": "dash"},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0.0, mux_knm],
            y=[0.0, muy_knm],
            mode="lines+markers",
            name="demand",
            line={"color": "#dc2626", "width": 3},
            marker={"size": [7, 13], "color": ["#334155", "#dc2626"], "symbol": ["circle", "x"]},
            hovertemplate="Mux=%{x:.1f} kN-m<br>Muy=%{y:.1f} kN-m<extra></extra>",
        )
    )
    fig.add_annotation(
        x=mux_knm,
        y=muy_knm,
        text=f"U = {check.load_contour_ratio:.3f}",
        showarrow=True,
        arrowhead=2,
        ax=28 if mux_knm <= 0 else -28,
        ay=28 if muy_knm <= 0 else -28,
        bgcolor="rgba(255,255,255,0.82)",
    )
    pad = 1.16 * max(cap_x, cap_y, abs(mux_knm), abs(muy_knm), 1.0)
    fig.update_layout(
        title={"text": f"Biaxial load contour at Pu = {check.axial_ratio * check.phi_pmax_kn:,.0f} kN", "x": 0.02, "xanchor": "left"},
        height=500,
        margin={"l": 24, "r": 24, "t": 54, "b": 24},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={"orientation": "h", "y": 1.02, "x": 0.52, "xanchor": "center"},
        font={"family": "Arial, sans-serif", "size": 13, "color": "#172033"},
    )
    fig.update_xaxes(title="Mux (kN-m)", range=[-pad, pad], gridcolor=COLORS["grid"], zeroline=True, zerolinecolor="#111827")
    fig.update_yaxes(
        title="Muy (kN-m)",
        range=[-pad, pad],
        gridcolor=COLORS["grid"],
        zeroline=True,
        zerolinecolor="#111827",
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def pmm_slice_plot(check: SectionCheck, mux_knm: float, muy_knm: float, pu_kn: float) -> go.Figure:
    points = check.pmm_slice
    fig = go.Figure()
    if not points:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="No PMM slice is available at this Pu level.",
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": "#64748b", "size": 14},
        )
        fig.update_layout(height=460, paper_bgcolor="white", plot_bgcolor="white")
        return fig

    contour_x = [point["mx_knm"] for point in points] + [points[0]["mx_knm"]]
    contour_y = [point["my_knm"] for point in points] + [points[0]["my_knm"]]
    fig.add_trace(
        go.Scatter(
            x=contour_x,
            y=contour_y,
            mode="lines",
            fill="toself",
            fillcolor="rgba(14, 165, 233, 0.10)",
            name="PMM slice",
            line={"color": "#0284c7", "width": 3},
            hovertemplate="Mux=%{x:.1f} kN-m<br>Muy=%{y:.1f} kN-m<extra></extra>",
        )
    )
    demand_color = "#0f766e" if check.pmm_ratio <= 1.0 else "#dc2626"
    fig.add_trace(
        go.Scatter(
            x=[0.0, mux_knm],
            y=[0.0, muy_knm],
            mode="lines+markers",
            name="demand",
            line={"color": demand_color, "width": 3},
            marker={"size": [7, 13], "color": ["#334155", demand_color], "symbol": ["circle", "x"]},
            hovertemplate="Mux=%{x:.1f} kN-m<br>Muy=%{y:.1f} kN-m<extra></extra>",
        )
    )
    fig.add_annotation(
        x=mux_knm,
        y=muy_knm,
        text=f"PMM U = {check.pmm_ratio:.3f}",
        showarrow=True,
        arrowhead=2,
        ax=28 if mux_knm <= 0 else -28,
        ay=28 if muy_knm <= 0 else -28,
        bgcolor="rgba(255,255,255,0.82)",
    )
    pad = 1.16 * max(
        max(abs(value) for value in contour_x),
        max(abs(value) for value in contour_y),
        abs(mux_knm),
        abs(muy_knm),
        1.0,
    )
    fig.update_layout(
        title={"text": f"PMM Mux-Muy slice at Pu = {pu_kn:,.0f} kN", "x": 0.02, "xanchor": "left"},
        height=500,
        margin={"l": 24, "r": 24, "t": 54, "b": 24},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={"orientation": "h", "y": 1.02, "x": 0.52, "xanchor": "center"},
        font={"family": "Arial, sans-serif", "size": 13, "color": "#172033"},
    )
    fig.update_xaxes(title="Mux (kN-m)", range=[-pad, pad], gridcolor=COLORS["grid"], zeroline=True, zerolinecolor="#111827")
    fig.update_yaxes(
        title="Muy (kN-m)",
        range=[-pad, pad],
        gridcolor=COLORS["grid"],
        zeroline=True,
        zerolinecolor="#111827",
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def pmm_surface_plot(check: SectionCheck, mux_knm: float, muy_knm: float, pu_kn: float) -> go.Figure:
    surface = check.pmm_surface
    fig = go.Figure()
    if not surface.get("mx") or not check.pmm_slice:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="Select PMM surface method to generate the 3D interaction surface.",
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": "#64748b", "size": 14},
        )
        fig.update_layout(height=560, paper_bgcolor="white", plot_bgcolor="white")
        return fig

    fig.add_trace(
        go.Surface(
            x=surface["mx"],
            y=surface["my"],
            z=surface["p"],
            name="PMM surface",
            colorscale="Blues",
            opacity=0.42,
            showscale=False,
            hovertemplate="Mux=%{x:.0f} kN-m<br>Muy=%{y:.0f} kN-m<br>Pu=%{z:.0f} kN<extra></extra>",
        )
    )
    slice_x = [point["mx_knm"] for point in check.pmm_slice] + [check.pmm_slice[0]["mx_knm"]]
    slice_y = [point["my_knm"] for point in check.pmm_slice] + [check.pmm_slice[0]["my_knm"]]
    slice_z = [pu_kn for _ in slice_x]
    fig.add_trace(
        go.Scatter3d(
            x=slice_x,
            y=slice_y,
            z=slice_z,
            mode="lines",
            name="current Pu slice",
            line={"color": "#0284c7", "width": 6},
            hovertemplate="Mux=%{x:.0f} kN-m<br>Muy=%{y:.0f} kN-m<br>Pu=%{z:.0f} kN<extra></extra>",
        )
    )
    demand_color = "#0f766e" if check.pmm_ratio <= 1.0 else "#dc2626"
    fig.add_trace(
        go.Scatter3d(
            x=[0.0, mux_knm],
            y=[0.0, muy_knm],
            z=[pu_kn, pu_kn],
            mode="lines+markers",
            name="demand vector",
            line={"color": demand_color, "width": 6},
            marker={"size": [3, 7], "color": ["#334155", demand_color], "symbol": "circle"},
            hovertemplate="Mux=%{x:.0f} kN-m<br>Muy=%{y:.0f} kN-m<br>Pu=%{z:.0f} kN<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[mux_knm],
            y=[muy_knm],
            z=[pu_kn],
            mode="markers+text",
            name="load point",
            marker={"size": 7, "color": demand_color},
            text=[f"U={check.pmm_ratio:.3f}"],
            textposition="top center",
            hovertemplate="Mux=%{x:.0f} kN-m<br>Muy=%{y:.0f} kN-m<br>Pu=%{z:.0f} kN<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": "3D PMM interaction surface with load point", "x": 0.02, "xanchor": "left"},
        height=620,
        margin={"l": 0, "r": 0, "t": 54, "b": 0},
        paper_bgcolor="white",
        font={"family": "Arial, sans-serif", "size": 13, "color": "#172033"},
        legend={"orientation": "h", "y": 1.02, "x": 0.5, "xanchor": "center"},
        scene={
            "xaxis": {"title": "Mux (kN-m)", "gridcolor": COLORS["grid"]},
            "yaxis": {"title": "Muy (kN-m)", "gridcolor": COLORS["grid"]},
            "zaxis": {"title": "Pu (kN)", "gridcolor": COLORS["grid"]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.45, "y": -1.55, "z": 1.05}},
        },
    )
    return fig


st.set_page_config(
    page_title="RC Bridge Abutment ULS",
    page_icon=":material/account_tree:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #64748b;
        --line: #d8dee7;
        --panel: #f8fafc;
        --ok: #0f766e;
        --ng: #b91c1c;
    }
    .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; }
    h1 { letter-spacing: 0; font-size: 2.05rem; }
    h2, h3 { letter-spacing: 0; }
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.35rem;
        border-bottom: 1px solid var(--line);
    }
    [data-testid="stTabs"] [role="tab"] {
        min-height: 3rem;
        padding: 0.65rem 1.0rem;
        border: 1px solid transparent;
        border-radius: 8px 8px 0 0;
        color: #334155;
    }
    [data-testid="stTabs"] [role="tab"] p {
        font-size: 1.30rem;
        font-weight: 700;
        line-height: 1.2;
    }
    [data-testid="stTabs"] [role="tab"]:hover {
        background: #f8fafc;
        border-color: #e2e8f0;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: #fff1f2;
        border-color: #fecdd3;
        box-shadow: inset 0 -3px 0 #ef4444;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] p {
        color: #e11d48;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border-radius: 8px;
        padding: 0.45rem 0.7rem;
        font-weight: 700;
        border: 1px solid var(--line);
        background: #ffffff;
    }
    .status-ok { color: var(--ok); border-color: #99f6e4; background: #f0fdfa; }
    .status-ng { color: var(--ng); border-color: #fecaca; background: #fef2f2; }
    .small-note { color: var(--muted); font-size: 0.92rem; line-height: 1.45; }
    </style>
    """,
    unsafe_allow_html=True,
)


def default_bearings(
    bearings_per_row: int,
    row_count: int,
    width_x_mm: float,
    height_z_mm: float,
    row_spacing_y_mm: float,
) -> pd.DataFrame:
    spacing = width_x_mm / (bearings_per_row + 1)
    y_positions = [0.0] if row_count == 1 else [row_spacing_y_mm / 2.0, -row_spacing_y_mm / 2.0]
    rows = []
    for row_index, y in enumerate(y_positions):
        row_label = "A" if row_count == 2 and row_index == 0 else "B" if row_count == 2 else ""
        for index in range(bearings_per_row):
            x = -width_x_mm / 2.0 + spacing * (index + 1)
            bearing_number = row_index * bearings_per_row + index + 1
            rows.append(
                {
                    "name": f"{row_label}{index + 1}" if row_count == 2 else f"B{bearing_number}",
                    "x_mm": round(x, 0),
                    "y_mm": round(y, 0),
                    "z_mm": height_z_mm,
                    "Pu_x_kN": 0.0,
                    "Pu_y_kN": 0.0,
                    "Pu_z_kN": 1200.0,
                    "Mu_x_kNm": 0.0,
                    "Mu_y_kNm": 0.0,
                }
            )
    return pd.DataFrame(rows)


def clean_bearings(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "name",
        "x_mm",
        "y_mm",
        "z_mm",
        "Pu_x_kN",
        "Pu_y_kN",
        "Pu_z_kN",
        "Mu_x_kNm",
        "Mu_y_kNm",
    ]
    for column in columns:
        if column not in df.columns:
            df[column] = "" if column == "name" else 0.0
    cleaned = df[columns].copy()
    cleaned["name"] = cleaned["name"].fillna("").astype(str)
    for column in columns[1:]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(0.0)
    return cleaned


PROJECT_FILE_VERSION = 1
PROJECT_SETTING_DEFAULTS = {
    "code_choice": "ACI 318 style",
    "fc_mpa": 35.0,
    "es_mpa": 200000.0,
    "phi_compression": 0.65,
    "phi_flexure": 0.90,
    "axial_cap_factor": 0.80,
    "eps_cu": 0.0030,
    "biaxial_method": "Load contour",
    "load_contour_alpha": 1.50,
    "structure_type": "Abutment with backfill",
    "width_x_mm": 9000.0,
    "depth_y_mm": 1200.0,
    "height_z_mm": 4500.0,
    "bearing_size_mm": 250.0,
    "bearing_size_x_mm": 250.0,
    "bearing_size_y_mm": 250.0,
    "pilecap_overhang_mm": 500.0,
    "pilecap_thickness_mm": 1500.0,
    "concrete_unit_weight_kn_m3": 24.0,
    "include_earth_pressure": True,
    "include_backfill_ev": True,
    "include_approach_slab_reaction": True,
    "earth_load_code": "AASHTO LRFD Strength I",
    "earth_backfill_height_m": 4.5,
    "backfill_ev_width_m": 0.50,
    "backfill_ev_height_m": 4.5,
    "backfill_ev_factor": BACKFILL_EV_FACTOR,
    "approach_slab_length_y_m": 10.0,
    "approach_slab_width_x_m": 9.0,
    "approach_slab_thickness_m": 0.25,
    "approach_slab_seat_length_m": 0.30,
    "approach_slab_superimposed_dead_kpa": 3.0,
    "approach_slab_gap_preset": "2.0 m - moderate settlement or local void",
    "approach_slab_gap_custom_m": 2.0,
    "approach_slab_ll_preset": "Road - AASHTO HL-93 lane load UDL only (3.1 kPa over 3 m lane)",
    "approach_slab_ll_q_kpa": 3.10,
    "approach_slab_ll_loaded_width_m": 3.0,
    "approach_slab_ll_line_load_kn_m": 80.0,
    "approach_slab_track_count": 1,
    "approach_slab_point_load_kn": 0.0,
    "approach_slab_point_load_y_m": 0.0,
    "approach_slab_factor_code": "AASHTO LRFD Strength I",
    "approach_slab_dc_factor": 1.25,
    "approach_slab_dw_factor": 1.50,
    "approach_slab_ll_factor": 1.75,
    "approach_slab_ll_replaces_surcharge": True,
    "soil_preset": "Thailand general backfill",
    "earth_gamma_soil_custom_kn_m3": 18.0,
    "earth_phi_custom_deg": 30.0,
    "earth_pressure_mode": "Active (Ka)",
    "en_traffic_category": "Road traffic (gamma_Q = 1.35)",
    "custom_eh_factor": 1.50,
    "custom_q_other_factor": 1.50,
    "custom_live_surcharge_factor": 1.75,
    "custom_gamma_tan_phi": 1.00,
    "q_other_preset": "None - traffic handled by live surcharge (0 kPa)",
    "q_other_custom_kpa": 10.0,
    "traffic_surcharge_preset": "Road bridge - AASHTO auto h_eq",
    "traffic_custom_h_eq_m": 0.75,
    "traffic_custom_q_kpa": 50.0,
    "earth_pressure_direction": "+y",
    "stem_bar_dia_mm": 20.0,
    "cover_mm": 75.0,
    "min_clear_spacing_mm": 100.0,
    "max_spacing_advisory_mm": 450.0,
    "enforce_max_spacing": True,
    "mode": "Auto design",
    "rho_min_percent": 0.25,
    "rho_max_percent": 4.00,
    "bar_dia_mm": 25.0,
    "bars_x_face": 12,
    "bars_y_face": 3,
    "shear_tie_dia_mm": 12.0,
    "shear_tie_legs": 2,
    "shear_spacing_provided_mm": 200.0,
    "dia_options": [20.0, 25.0, 28.0, 32.0],
    "bearing_rows": 1,
    "bearings_per_row": 4,
    "row_spacing_y_mm": 600.0,
    "strip_width_mode": "Auto effective strip",
    "manual_strip_width_mm": None,
    "strength_bearing_names": [],
}


def _session_project_value(key: str):
    return st.session_state.get(key, PROJECT_SETTING_DEFAULTS[key])


def project_payload_json() -> str:
    settings = {key: _session_project_value(key) for key in PROJECT_SETTING_DEFAULTS}
    bearing_rows = int(settings["bearing_rows"])
    bearings_per_row = int(settings["bearings_per_row"])
    width_x = float(settings["width_x_mm"])
    height_z = float(settings["height_z_mm"])
    row_spacing = float(settings["row_spacing_y_mm"])
    default_table = default_bearings(bearings_per_row, bearing_rows, width_x, height_z, row_spacing)
    bearing_table = clean_bearings(pd.DataFrame(st.session_state.get("bearing_table", default_table)))
    payload = {
        "app": "RC Bridge Abutment ULS Designer",
        "version": PROJECT_FILE_VERSION,
        "settings": settings,
        "bearing_table": bearing_table.to_dict("records"),
    }
    return json.dumps(payload, indent=2)


def apply_project_payload(payload: dict) -> None:
    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("Project file is missing the settings object.")

    if "bearing_size_x_mm" not in settings and "bearing_size_mm" in settings:
        settings["bearing_size_x_mm"] = settings["bearing_size_mm"]
    if "bearing_size_y_mm" not in settings and "bearing_size_mm" in settings:
        settings["bearing_size_y_mm"] = settings["bearing_size_mm"]

    for key in PROJECT_SETTING_DEFAULTS:
        if key in settings:
            if key == "manual_strip_width_mm" and settings[key] is None:
                st.session_state.pop(key, None)
                continue
            st.session_state[key] = settings[key]

    bearing_table = payload.get("bearing_table")
    if bearing_table is not None:
        st.session_state.bearing_table = clean_bearings(pd.DataFrame(bearing_table))
        st.session_state.pending_bearing_table = st.session_state.bearing_table.copy()
        st.session_state.bearing_table_dirty = False

    st.session_state.pop("bearing_load_editor", None)


def sync_bearing_editor(editor_key: str) -> None:
    editor_state = st.session_state.get(editor_key)
    if not isinstance(editor_state, dict):
        return
    base_table = st.session_state.get("pending_bearing_table")
    if base_table is None:
        base_table = st.session_state.get("bearing_table")
    if base_table is None:
        return

    updated = clean_bearings(pd.DataFrame(base_table))
    for row_index, changes in editor_state.get("edited_rows", {}).items():
        index = int(row_index)
        if index >= len(updated):
            continue
        for column, value in changes.items():
            if column in updated.columns:
                updated.at[index, column] = value
    st.session_state.pending_bearing_table = clean_bearings(updated)
    st.session_state.bearing_table_dirty = True


def committed_editor_table(
    returned_table: pd.DataFrame,
    base_table: pd.DataFrame,
    editor_key: str,
) -> pd.DataFrame:
    committed = clean_bearings(pd.DataFrame(returned_table))
    editor_state = st.session_state.get(editor_key)
    if not isinstance(editor_state, dict):
        return committed

    edited_rows = editor_state.get("edited_rows", {})
    if not edited_rows:
        return committed

    fallback = clean_bearings(pd.DataFrame(base_table))
    for row_index, changes in edited_rows.items():
        index = int(row_index)
        if index >= len(fallback):
            continue
        for column, value in changes.items():
            if column in fallback.columns:
                fallback.at[index, column] = value
    return clean_bearings(fallback)


def status_html(status: str, ratio: float) -> str:
    klass = "status-ok" if status == "OK" else "status-ng"
    label = "PASS" if status == "OK" else "FAIL"
    return f'<span class="status-pill {klass}">{label} &nbsp; U = {ratio:.3f}</span>'


def metric_row(resultant, check, design_width_x_mm: float | None = None):
    cols = st.columns(6)
    cols[0].metric("Pu compression", f"{resultant.pu_kn:,.0f} kN")
    cols[1].metric("Vx resultant", f"{resultant.vx_kn:,.0f} kN")
    cols[2].metric("Vy resultant", f"{resultant.vy_kn:,.0f} kN")
    cols[3].metric("Mux design", f"{resultant.design_mux_knm:,.0f} kN-m")
    cols[4].metric("Muy design", f"{resultant.design_muy_knm:,.0f} kN-m")
    cols[5].metric("Tz resultant", f"{resultant.torsion_z_knm:,.0f} kN-m")

    if check is not None:
        cols = st.columns(6)
        cols[0].metric("phi Pmax", f"{check.phi_pmax_kn:,.0f} kN")
        cols[1].metric("phi Mnx at Pu", f"{check.phi_mnx_at_pu_knm:,.0f} kN-m")
        cols[2].metric("phi Mny at Pu", f"{check.phi_mny_at_pu_knm:,.0f} kN-m")
        cols[3].metric("As provided", f"{check.as_total_mm2:,.0f} mm2")
        cols[4].metric("rho", f"{check.rho_percent:.3f} %")
        cols[5].metric(
            "design width",
            f"{design_width_x_mm if design_width_x_mm is not None else check.width_x_mm:,.0f} mm",
            delta=f"max c/c {check.max_center_spacing_mm:,.0f} mm",
            delta_color="off",
        )


st.title("RC Bridge Abutment / Wall Pier ULS Designer")
st.caption("Bearing and earth-pressure ULS resultants, PMM checks, stem design, and pile-cap interface summaries.")

with st.sidebar:
    st.markdown("### 💾 Save / Load Design")
    project_message = st.session_state.pop("project_file_message", None)
    if project_message:
        st.success(project_message)
    save_col, open_col = st.columns([1.0, 1.0])
    with save_col:
        st.download_button(
            "💾 Save",
            data=project_payload_json(),
            file_name="rc_abutment_uls_project.json",
            mime="application/json",
            width="stretch",
            key="project_save_button",
        )
    with open_col:
        uploaded_project = st.file_uploader(
            "📂 Open File",
            type=["json"],
            key="project_file_upload",
            help="Upload a JSON file previously saved from this app. It will open automatically.",
        )
    if uploaded_project is not None:
        uploaded_bytes = uploaded_project.getvalue()
        uploaded_text = uploaded_bytes.decode("utf-8")
        upload_signature = f"{uploaded_project.name}:{len(uploaded_bytes)}:{uploaded_text[:80]}:{uploaded_text[-80:]}"
        if upload_signature != st.session_state.get("project_loaded_upload_signature"):
            try:
                payload = json.loads(uploaded_text)
                apply_project_payload(payload)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not open project file: {exc}")
            else:
                st.session_state.project_loaded_upload_signature = upload_signature
                st.session_state.project_file_message = "Project file loaded."
                st.rerun()
    st.caption(
        "💡 ไฟล์จะถูกบันทึกที่ Downloads folder หากต้องการเลือก folder เอง "
        "ให้เปิด Ask where to save ในการตั้งค่า Browser"
    )

    st.header("Design Basis")
    code_choice = st.selectbox(
        "Code assumptions",
        ["ACI 318 style", "AASHTO LRFD style"],
        key="code_choice",
        help="Resistance-factor defaults are editable below. Always verify against the governing project code edition.",
    )
    structure_type = st.selectbox(
        "Structure type",
        STRUCTURE_TYPES,
        key="structure_type",
        help="Abutment mode can include earth pressure. Wall Pier mode excludes backfill earth pressure.",
    )
    is_abutment_mode = structure_type == "Abutment with backfill"
    if is_abutment_mode:
        st.caption("Abutment mode: bearing ULS loads are kept as-entered; app-generated earth pressure is factored separately.")
    else:
        st.caption("Wall Pier mode: earth pressure is excluded; use bearing ULS loads and self-weight only.")
    base_params = default_code_parameters(code_choice)

    fc_mpa = st.number_input("f'c (MPa)", min_value=15.0, max_value=100.0, value=35.0, step=1.0, key="fc_mpa")
    es_mpa = st.number_input("Es (MPa)", min_value=180000.0, max_value=220000.0, value=200000.0, step=5000.0, key="es_mpa")
    st.caption("Rebar fy is assigned automatically: DB12/16/20/25/28 = 390 MPa, DB32 = 490 MPa.")

    with st.expander("Resistance factor settings", expanded=False):
        phi_compression = st.number_input(
            "phi compression",
            min_value=0.40,
            max_value=0.95,
            value=float(base_params.phi_compression),
            step=0.01,
            key="phi_compression",
        )
        phi_flexure = st.number_input(
            "phi flexure / tension",
            min_value=0.40,
            max_value=0.95,
            value=float(base_params.phi_flexure),
            step=0.01,
            key="phi_flexure",
        )
        axial_cap_factor = st.number_input(
            "axial cap factor",
            min_value=0.50,
            max_value=1.00,
            value=float(base_params.axial_cap_factor),
            step=0.01,
            key="axial_cap_factor",
        )
        eps_cu = st.number_input(
            "concrete ultimate strain",
            min_value=0.0020,
            max_value=0.0040,
            value=0.0030,
            step=0.0001,
            format="%.4f",
            key="eps_cu",
        )

    params = replace(
        base_params,
        phi_compression=phi_compression,
        phi_flexure=phi_flexure,
        axial_cap_factor=axial_cap_factor,
        eps_cu=eps_cu,
    )

    biaxial_method = st.radio("Biaxial check method", BIAXIAL_METHODS, horizontal=True, key="biaxial_method")
    load_contour_alpha = st.number_input(
        "load contour alpha",
        min_value=1.00,
        max_value=2.00,
        value=1.50,
        step=0.05,
        disabled=biaxial_method != "Load contour",
        key="load_contour_alpha",
        help="alpha=1.0 equals the conservative linear interaction. Larger alpha gives a rounded load contour.",
    )
    with st.expander("Alpha guidance and code reference", expanded=False):
        st.markdown(ALPHA_GUIDANCE_MD)
    if biaxial_method == PMM_METHOD:
        st.caption(
            "PMM surface uses rotated neutral-axis strain compatibility and plots the 3D Pu-Mux-Muy surface. "
            "Auto design uses a fast preliminary screen, then checks promising layouts with PMM."
        )

    st.header("Geometry")
    width_x_mm = st.number_input("Abutment width along x (mm)", min_value=800.0, value=9000.0, step=100.0, key="width_x_mm")
    depth_y_mm = st.number_input("Abutment thickness along y (mm)", min_value=300.0, value=1200.0, step=50.0, key="depth_y_mm")
    height_z_mm = st.number_input("Bearing level height z (mm)", min_value=500.0, value=4500.0, step=100.0, key="height_z_mm")
    legacy_bearing_size_mm = float(st.session_state.get("bearing_size_mm", 250.0))
    bearing_cols = st.columns(2)
    with bearing_cols[0]:
        bearing_size_x_mm = st.number_input(
            "Bearing size along x (mm)",
            min_value=100.0,
            value=float(st.session_state.get("bearing_size_x_mm", legacy_bearing_size_mm)),
            step=25.0,
            key="bearing_size_x_mm",
        )
    with bearing_cols[1]:
        bearing_size_y_mm = st.number_input(
            "Bearing size along y (mm)",
            min_value=100.0,
            value=float(st.session_state.get("bearing_size_y_mm", legacy_bearing_size_mm)),
            step=25.0,
            key="bearing_size_y_mm",
        )
    st.session_state.bearing_size_mm = bearing_size_x_mm
    pilecap_overhang_mm = st.number_input("Pile cap overhang each side (mm)", min_value=0.0, value=500.0, step=50.0, key="pilecap_overhang_mm")
    pilecap_thickness_mm = st.number_input("Pile cap display thickness (mm)", min_value=300.0, value=1500.0, step=100.0, key="pilecap_thickness_mm")
    concrete_unit_weight_kn_m3 = st.number_input(
        "Concrete unit weight (kN/m3)",
        min_value=18.0,
        max_value=30.0,
        value=24.0,
        step=0.5,
        key="concrete_unit_weight_kn_m3",
    )

    earth_pressure_factor_candidates: list[EarthPressureFactors] = []
    if is_abutment_mode:
        st.header("Earth Pressure")
        include_earth_pressure = st.checkbox(
            "Include earth pressure in section design, pile cap summary, and stem design",
            value=True,
            key="include_earth_pressure",
        )
        if st.session_state.get("earth_load_code") == "AASHTO LRFD":
            st.session_state.earth_load_code = "AASHTO LRFD Strength I"
        earth_load_code = st.selectbox(
            "Generated earth pressure ULS code",
            EARTH_LOAD_CODES,
            key="earth_load_code",
            help="Bearing loads are already ULS and are not factored again. This code only factors app-generated EH/EV/surcharge loads.",
        )
        earth_backfill_height_m = st.number_input(
            "Backfill height H (m)",
            min_value=0.10,
            max_value=30.0,
            value=4.50,
            step=0.10,
            key="earth_backfill_height_m",
            help="Height of retained backfill producing lateral pressure on the abutment.",
        )
        soil_preset = st.selectbox(
            "Soil preset",
            list(SOIL_PRESETS),
            key="soil_preset",
            help="Default Thailand general backfill is intended only for preliminary use when geotechnical data is unavailable.",
        )
        if soil_preset == "Custom":
            earth_gamma_soil_kn_m3 = st.number_input(
                "gamma_soil (kN/m3)",
                min_value=10.0,
                max_value=25.0,
                value=18.0,
                step=0.5,
                key="earth_gamma_soil_custom_kn_m3",
            )
            earth_phi_deg = st.number_input(
                "friction angle phi (deg)",
                min_value=15.0,
                max_value=45.0,
                value=30.0,
                step=1.0,
                key="earth_phi_custom_deg",
            )
        else:
            soil_values = SOIL_PRESETS[soil_preset]
            earth_gamma_soil_kn_m3 = float(soil_values["gamma_soil_kn_m3"])
            earth_phi_deg = float(soil_values["phi_deg"])
            st.caption(
                f"{soil_values['note']} gamma_soil = {earth_gamma_soil_kn_m3:.1f} kN/m3, "
                f"phi = {earth_phi_deg:.0f} deg."
            )
        earth_pressure_mode = st.selectbox(
            "Pressure mode",
            ["Active (Ka)", "At-rest (K0)"],
            key="earth_pressure_mode",
            help="Active assumes the abutment can move slightly away from soil. At-rest is for restrained/stiffer cases.",
        )
        q_other_preset = st.selectbox(
            "q_other permanent surcharge",
            list(Q_OTHER_PRESETS_KPA),
            key="q_other_preset",
            help="Use for pavement, ballast, storage, or other non-traffic surcharge on the backfill. Do not repeat traffic here.",
        )
        if q_other_preset == "Custom":
            q_other_kpa = st.number_input(
                "custom q_other (kPa)",
                min_value=0.0,
                max_value=200.0,
                value=10.0,
                step=1.0,
                key="q_other_custom_kpa",
            )
        else:
            q_other_kpa = float(Q_OTHER_PRESETS_KPA[q_other_preset])

        traffic_surcharge_preset = st.selectbox(
            "Traffic live load surcharge",
            list(TRAFFIC_SURCHARGE_PRESETS),
            key="traffic_surcharge_preset",
            help="Road bridge uses AASHTO-style auto h_eq. Railway presets use equivalent uniform q_LS.",
        )
        traffic_surcharge = TRAFFIC_SURCHARGE_PRESETS[traffic_surcharge_preset]
        if traffic_surcharge["type"] == "road_auto":
            traffic_h_eq_m = aashto_road_h_eq_m(earth_backfill_height_m)
            traffic_q_kpa = earth_gamma_soil_kn_m3 * traffic_h_eq_m
        elif traffic_surcharge["type"] == "q":
            traffic_q_kpa = float(traffic_surcharge["q_kpa"])
            traffic_h_eq_m = traffic_q_kpa / max(earth_gamma_soil_kn_m3, 1e-9)
        elif traffic_surcharge["type"] == "h_eq":
            traffic_h_eq_m = st.number_input(
                "custom h_eq (m)",
                min_value=0.0,
                max_value=10.0,
                value=0.75,
                step=0.05,
                key="traffic_custom_h_eq_m",
            )
            traffic_q_kpa = earth_gamma_soil_kn_m3 * traffic_h_eq_m
        else:
            traffic_q_kpa = st.number_input(
                "custom q_LS (kPa)",
                min_value=0.0,
                max_value=300.0,
                value=50.0,
                step=5.0,
                key="traffic_custom_q_kpa",
            )
            traffic_h_eq_m = traffic_q_kpa / max(earth_gamma_soil_kn_m3, 1e-9)

        en_traffic_category = st.selectbox(
            "EN traffic factor category",
            list(EN_TRAFFIC_CATEGORIES),
            key="en_traffic_category",
            disabled=not earth_load_code.startswith("EN"),
            help="Used only for EN DA1 Combination 1 traffic surcharge factor. Verify with the project National Annex.",
        )
        with st.expander("Custom earth pressure factors", expanded=earth_load_code == "Custom factors"):
            custom_eh_factor = st.number_input("custom gamma_EH", min_value=0.0, max_value=3.0, value=1.50, step=0.05, key="custom_eh_factor")
            custom_q_other_factor = st.number_input(
                "custom gamma_q_other",
                min_value=0.0,
                max_value=3.0,
                value=1.50,
                step=0.05,
                key="custom_q_other_factor",
            )
            custom_live_factor = st.number_input(
                "custom gamma_LS",
                min_value=0.0,
                max_value=3.0,
                value=1.75,
                step=0.05,
                key="custom_live_surcharge_factor",
            )
            custom_gamma_tan_phi = st.number_input(
                "custom gamma_tan_phi",
                min_value=0.50,
                max_value=2.00,
                value=1.00,
                step=0.05,
                key="custom_gamma_tan_phi",
                help="Use 1.00 for no soil strength reduction. EN DA1 C2 typically uses 1.25.",
            )
        earth_pressure_direction = st.selectbox(
            "Earth pressure direction",
            ["+y", "-y"],
            key="earth_pressure_direction",
            help="Controls the sign of Vy and Mu_x in the pile cap force summary. Backfill EV is placed on the opposite side of this pressure direction.",
        )
        with st.expander("Backfill vertical load on pile cap (EV)", expanded=True):
            include_backfill_ev = st.checkbox(
                "Include vertical backfill weight on pile cap / heel",
                value=True,
                key="include_backfill_ev",
            )
            backfill_ev_width_m = st.number_input(
                "b_EV: backfill width supported by pile cap (m)",
                min_value=0.0,
                max_value=30.0,
                value=float(st.session_state.get("backfill_ev_width_m", max(pilecap_overhang_mm / 1000.0, 0.0))),
                step=0.10,
                key="backfill_ev_width_m",
                help="Measure only the heel/backfill side from the back face of the abutment/stem to the rear edge of pile cap. Do not include the front overhang unless it also supports backfill.",
            )
            backfill_ev_height_m = st.number_input(
                "h_EV: average fill height over that area (m)",
                min_value=0.0,
                max_value=30.0,
                value=float(st.session_state.get("backfill_ev_height_m", earth_backfill_height_m)),
                step=0.10,
                key="backfill_ev_height_m",
                help="Use the average vertical soil height bearing on the pile cap/heel. It may differ from lateral-pressure H if elevations differ.",
            )
            backfill_ev_factor = st.number_input(
                "gamma_EV",
                min_value=0.0,
                max_value=3.0,
                value=float(st.session_state.get("backfill_ev_factor", BACKFILL_EV_FACTOR)),
                step=0.05,
                key="backfill_ev_factor",
                help="Typical AASHTO LRFD maximum EV factor for retaining walls/abutments is 1.35. Use the governing project load combination.",
            )
            st.caption(
                "Preliminary EV area = Lx x b_EV. The centroid is placed at the middle of b_EV on the backfill side of the abutment."
            )
        with st.expander("Approach slab reaction - soil supported mode", expanded=True):
            include_approach_slab_reaction = st.checkbox(
                "Include approach slab reaction on abutment / pile cap",
                value=True,
                key="include_approach_slab_reaction",
            )
            approach_slab_length_y_m = st.number_input(
                "Approach slab length along y (m)",
                min_value=0.0,
                max_value=50.0,
                value=float(st.session_state.get("approach_slab_length_y_m", 10.0)),
                step=0.10,
                key="approach_slab_length_y_m",
            )
            approach_slab_width_x_m = st.number_input(
                "Approach slab width along x (m)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.get("approach_slab_width_x_m", width_x_mm / 1000.0)),
                step=0.10,
                key="approach_slab_width_x_m",
                help="Usually equals abutment width, but may be adjusted for staged construction or partial slab width.",
            )
            approach_slab_thickness_m = st.number_input(
                "Approach slab thickness (m)",
                min_value=0.0,
                max_value=2.0,
                value=float(st.session_state.get("approach_slab_thickness_m", 0.25)),
                step=0.01,
                key="approach_slab_thickness_m",
            )
            approach_slab_seat_max_m = max(0.0, float(depth_y_mm) / 1000.0)
            approach_slab_seat_default_m = min(
                approach_slab_seat_max_m,
                max(0.0, float(st.session_state.get("approach_slab_seat_length_m", 0.30))),
            )
            approach_slab_seat_length_m = st.number_input(
                "Approach slab seat length on abutment (m)",
                min_value=0.0,
                max_value=approach_slab_seat_max_m,
                value=approach_slab_seat_default_m,
                step=0.05,
                key="approach_slab_seat_length_m",
                help=(
                    "Bearing length of the approach slab on the abutment top. "
                    "In this soil-supported mode it shifts y_AS and Mux; the reaction magnitude still follows l_gap."
                ),
            )
            approach_slab_superimposed_dead_kpa = st.number_input(
                "Superimposed dead load on slab q_DW (kPa)",
                min_value=0.0,
                max_value=200.0,
                value=float(st.session_state.get("approach_slab_superimposed_dead_kpa", 3.0)),
                step=0.5,
                key="approach_slab_superimposed_dead_kpa",
                help="Use for asphalt, waterproofing, track slab, ballast, rail components, or other permanent loads carried by the approach slab.",
            )
            approach_slab_gap_preset = st.selectbox(
                "Recommended effective unsupported length l_gap",
                APPROACH_SLAB_GAP_PRESETS,
                key="approach_slab_gap_preset",
                help="Mode 1 assumes only the slab length over the settlement gap bridges to the abutment. Soil-supported parts are not sent as abutment reaction.",
            )
            approach_slab_gap_custom_m = st.number_input(
                "custom l_gap (m)",
                min_value=0.0,
                max_value=max(50.0, float(approach_slab_length_y_m)),
                value=float(st.session_state.get("approach_slab_gap_custom_m", 2.0)),
                step=0.10,
                key="approach_slab_gap_custom_m",
                disabled=approach_slab_gap_preset != "Custom",
            )
            approach_slab_gap_m = approach_slab_gap_from_preset(
                approach_slab_gap_preset,
                approach_slab_length_y_m,
                approach_slab_gap_custom_m,
            )
            st.metric("l_gap used", f"{approach_slab_gap_m:,.2f} m")

            approach_slab_ll_preset = st.selectbox(
                "Approach slab live load model",
                list(APPROACH_SLAB_LIVE_LOAD_PRESETS),
                key="approach_slab_ll_preset",
                help="Presets are preliminary UDL components only. Add axle or concentrated reaction below when it governs.",
            )
            ll_preset = APPROACH_SLAB_LIVE_LOAD_PRESETS[approach_slab_ll_preset]
            ll_type = str(ll_preset["type"])
            if ll_type == "area":
                approach_slab_ll_q_kpa = float(ll_preset["q_kpa"])
                st.metric("q_LL preset", f"{approach_slab_ll_q_kpa:,.2f} kPa")
                approach_slab_ll_loaded_width_m = st.number_input(
                    "LL loaded width along x (m)",
                    min_value=0.0,
                    max_value=max(100.0, float(approach_slab_width_x_m)),
                    value=float(st.session_state.get("approach_slab_ll_loaded_width_m", ll_preset["default_width_m"])),
                    step=0.10,
                    key="approach_slab_ll_loaded_width_m",
                )
                approach_slab_ll_line_load_kn_m = 0.0
                approach_slab_track_count = 0
            elif ll_type == "area_custom":
                approach_slab_ll_q_kpa = st.number_input(
                    "q_LL on approach slab (kPa)",
                    min_value=0.0,
                    max_value=500.0,
                    value=float(st.session_state.get("approach_slab_ll_q_kpa", ll_preset["q_kpa"])),
                    step=0.5,
                    key="approach_slab_ll_q_kpa",
                )
                approach_slab_ll_loaded_width_m = st.number_input(
                    "LL loaded width along x (m)",
                    min_value=0.0,
                    max_value=max(100.0, float(approach_slab_width_x_m)),
                    value=float(st.session_state.get("approach_slab_ll_loaded_width_m", ll_preset["default_width_m"])),
                    step=0.10,
                    key="approach_slab_ll_loaded_width_m",
                )
                approach_slab_ll_line_load_kn_m = 0.0
                approach_slab_track_count = 0
            elif ll_type == "line":
                approach_slab_ll_line_load_kn_m = float(ll_preset["q_line_kn_m"])
                st.metric("q_LL line preset", f"{approach_slab_ll_line_load_kn_m:,.2f} kN/m/track")
                approach_slab_track_count = st.number_input(
                    "Number of loaded tracks",
                    min_value=0,
                    max_value=10,
                    value=int(st.session_state.get("approach_slab_track_count", 1)),
                    step=1,
                    key="approach_slab_track_count",
                )
                approach_slab_ll_q_kpa = 0.0
                approach_slab_ll_loaded_width_m = 0.0
            elif ll_type == "line_custom":
                approach_slab_ll_line_load_kn_m = st.number_input(
                    "q_LL line load per track (kN/m)",
                    min_value=0.0,
                    max_value=500.0,
                    value=float(st.session_state.get("approach_slab_ll_line_load_kn_m", ll_preset["q_line_kn_m"])),
                    step=5.0,
                    key="approach_slab_ll_line_load_kn_m",
                )
                approach_slab_track_count = st.number_input(
                    "Number of loaded tracks",
                    min_value=0,
                    max_value=10,
                    value=int(st.session_state.get("approach_slab_track_count", 1)),
                    step=1,
                    key="approach_slab_track_count",
                )
                approach_slab_ll_q_kpa = 0.0
                approach_slab_ll_loaded_width_m = 0.0
            else:
                approach_slab_ll_q_kpa = 0.0
                approach_slab_ll_loaded_width_m = 0.0
                approach_slab_ll_line_load_kn_m = 0.0
                approach_slab_track_count = 0
            approach_slab_point_load_kn = st.number_input(
                "Additional LL point/axle load in gap (kN)",
                min_value=0.0,
                max_value=5000.0,
                value=float(st.session_state.get("approach_slab_point_load_kn", 0.0)),
                step=10.0,
                key="approach_slab_point_load_kn",
                help="Optional total concentrated load within l_gap. Reaction uses P x (l_gap - y) / l_gap.",
            )
            approach_slab_point_load_y_m = st.number_input(
                "Point/axle load distance from abutment y (m)",
                min_value=0.0,
                max_value=max(50.0, float(approach_slab_length_y_m)),
                value=float(st.session_state.get("approach_slab_point_load_y_m", 0.0)),
                step=0.10,
                key="approach_slab_point_load_y_m",
            )
            approach_slab_factor_code = st.selectbox(
                "Approach slab load factor code",
                list(APPROACH_SLAB_FACTOR_PRESETS),
                key="approach_slab_factor_code",
                help="Select the load-factor set for approach slab reactions. Use Custom factors for project-specific combinations.",
            )
            factor_preset = APPROACH_SLAB_FACTOR_PRESETS[approach_slab_factor_code]
            factor_values = {
                "dc": float(factor_preset["dc"] if factor_preset["dc"] is not None else st.session_state.get("approach_slab_dc_factor", 1.25)),
                "dw": float(factor_preset["dw"] if factor_preset["dw"] is not None else st.session_state.get("approach_slab_dw_factor", 1.50)),
                "ll": float(factor_preset["ll"] if factor_preset["ll"] is not None else st.session_state.get("approach_slab_ll_factor", 1.75)),
            }
            if approach_slab_factor_code != "Custom factors":
                st.session_state.approach_slab_dc_factor = factor_values["dc"]
                st.session_state.approach_slab_dw_factor = factor_values["dw"]
                st.session_state.approach_slab_ll_factor = factor_values["ll"]
            factor_cols = st.columns(3)
            with factor_cols[0]:
                approach_slab_dc_factor = st.number_input(
                    "gamma_AS_DC",
                    min_value=0.0,
                    max_value=3.0,
                    value=factor_values["dc"],
                    step=0.05,
                    key="approach_slab_dc_factor",
                    disabled=approach_slab_factor_code != "Custom factors",
                )
            with factor_cols[1]:
                approach_slab_dw_factor = st.number_input(
                    "gamma_AS_DW",
                    min_value=0.0,
                    max_value=3.0,
                    value=factor_values["dw"],
                    step=0.05,
                    key="approach_slab_dw_factor",
                    disabled=approach_slab_factor_code != "Custom factors",
                )
            with factor_cols[2]:
                approach_slab_ll_factor = st.number_input(
                    "gamma_AS_LL",
                    min_value=0.0,
                    max_value=3.0,
                    value=factor_values["ll"],
                    step=0.05,
                    key="approach_slab_ll_factor",
                    disabled=approach_slab_factor_code != "Custom factors",
                )
            st.caption(
                f"{approach_slab_factor_code}: "
                f"gamma_DC = {approach_slab_dc_factor:.2f}, "
                f"gamma_DW = {approach_slab_dw_factor:.2f}, "
                f"gamma_LL = {approach_slab_ll_factor:.2f}."
            )
            approach_slab_ll_replaces_surcharge = st.checkbox(
                "Do not also apply traffic live load surcharge q_LS when approach slab LL reaction is included",
                value=True,
                key="approach_slab_ll_replaces_surcharge",
                help="Prevents double counting the same vehicle/train live load as both vertical slab reaction and lateral surcharge.",
            )
            if include_approach_slab_reaction and approach_slab_ll_replaces_surcharge and approach_slab_ll_preset != "None":
                traffic_q_kpa = 0.0
                traffic_h_eq_m = 0.0
                st.info("Traffic live load surcharge q_LS is set to 0.0 because approach slab LL reaction is included.")
        stem_bar_dia_mm = st.selectbox(
            "Stem vertical bar diameter for spacing guide",
            REBAR_DIAMETERS_MM,
            index=2,
            key="stem_bar_dia_mm",
            format_func=lambda dia: f"{rebar_label(dia)} fy={rebar_fy_mpa(dia):.0f} MPa",
            help="Used only for the preliminary 1 m stem strip spacing guide.",
        )
        earth_pressure_factor_candidates = earth_pressure_factor_sets(
            earth_load_code,
            earth_pressure_mode,
            earth_phi_deg,
            en_traffic_category,
            custom_eh_factor,
            custom_q_other_factor,
            custom_live_factor,
            custom_gamma_tan_phi,
        )
        controlling_factor = earth_pressure_factor_candidates[-1]
        earth_pressure_k = earth_pressure_coefficient(controlling_factor.phi_design_deg, earth_pressure_mode)
        st.caption(
            f"{earth_load_code}: K = {earth_pressure_k:.3f}, q_other = {q_other_kpa:.1f} kPa, "
            f"traffic h_eq = {traffic_h_eq_m:.2f} m, q_LS = {traffic_q_kpa:.1f} kPa."
        )
        with st.expander("Earth pressure quick guide", expanded=False):
            st.markdown(
                """
                `Active (Ka)` is for an abutment or wall that can move slightly away from the backfill.  
                `At-rest (K0)` is for restrained or very stiff cases and usually gives larger pressure.

                `q_other` is a permanent or non-traffic surcharge on the backfill, in kPa. Use it for pavement,
                ballast, storage, construction allowance, or project-specific uniform surcharge.

                `h_eq` is an equivalent height of soil for traffic live load surcharge. It is not an actual fill height.
                The app converts it to `q_LS = gamma_soil x h_eq`, then to lateral pressure `K x q_LS`.

                `EV` is the vertical backfill weight directly supported by the pile cap/heel. Use `b_EV` only for the
                backfill-side projection from the abutment/stem face to the pile cap edge, and use the average fill
                height above that area.

                Bearing table loads are assumed to be already factored ULS loads. The selected earth pressure code
                only factors app-generated EH, EV, q_other/ES, and traffic LS. The factored earth pressure is added to
                the strength-section design over the selected strip width, while EV is added to the pile-cap summary.
                Do not include the same traffic load in
                both `q_other` and traffic live load surcharge.
                """
            )
    else:
        include_earth_pressure = False
        include_backfill_ev = False
        include_approach_slab_reaction = False
        backfill_ev_width_m = 0.0
        backfill_ev_height_m = 0.0
        backfill_ev_factor = BACKFILL_EV_FACTOR
        approach_slab_length_y_m = 0.0
        approach_slab_width_x_m = 0.0
        approach_slab_thickness_m = 0.0
        approach_slab_seat_length_m = 0.0
        approach_slab_superimposed_dead_kpa = 0.0
        approach_slab_gap_m = 0.0
        approach_slab_ll_preset = "None"
        approach_slab_ll_q_kpa = 0.0
        approach_slab_ll_loaded_width_m = 0.0
        approach_slab_ll_line_load_kn_m = 0.0
        approach_slab_track_count = 0
        approach_slab_point_load_kn = 0.0
        approach_slab_point_load_y_m = 0.0
        approach_slab_factor_code = "AASHTO LRFD Strength I"
        approach_slab_dc_factor = 1.25
        approach_slab_dw_factor = 1.50
        approach_slab_ll_factor = 1.75
        approach_slab_ll_replaces_surcharge = False
        st.header("Earth Pressure")
        st.info("Wall Pier mode is selected, so backfill earth pressure inputs are hidden and excluded from the calculations.")

    st.header("Reinforcement")
    cover_mm = st.number_input("Clear cover to tie / outer bar (mm)", min_value=25.0, value=75.0, step=5.0, key="cover_mm")
    min_clear_spacing_mm = st.number_input("minimum clear bar spacing (mm)", min_value=25.0, value=100.0, step=25.0, key="min_clear_spacing_mm")
    default_max_spacing_mm = min(3.0 * depth_y_mm, 450.0)
    max_spacing_advisory_mm = st.number_input(
        "max bar spacing advisory, c/c (mm)",
        min_value=100.0,
        max_value=2000.0,
        value=float(default_max_spacing_mm),
        step=25.0,
        key="max_spacing_advisory_mm",
        help="Advisory default for wall/abutment-style distributed reinforcement: min(3t, 450 mm). Verify with the governing code edition and project specification.",
    )
    enforce_max_spacing = st.checkbox("enforce max spacing in auto design", value=True, key="enforce_max_spacing")
    mode = st.radio("Reinforcement mode", ["Auto design", "Manual check"], horizontal=True, key="mode")
    rho_min_percent = st.number_input("minimum rho for auto (%)", min_value=0.0, max_value=5.0, value=0.25, step=0.05, key="rho_min_percent")
    rho_max_percent = st.number_input("maximum rho for auto (%)", min_value=0.1, max_value=10.0, value=4.00, step=0.10, key="rho_max_percent")

    if mode == "Manual check":
        bar_dia_mm = st.selectbox(
            "bar diameter",
            REBAR_DIAMETERS_MM,
            index=3,
            key="bar_dia_mm",
            format_func=lambda dia: f"{rebar_label(dia)}  fy={rebar_fy_mpa(dia):.0f} MPa",
        )
        selected_fy_mpa = rebar_fy_mpa(float(bar_dia_mm))
        bars_x_face = st.number_input("bars along x on top/bottom faces", min_value=2, max_value=40, value=12, step=1, key="bars_x_face")
        bars_y_face = st.number_input("bars along y on left/right faces", min_value=2, max_value=24, value=3, step=1, key="bars_y_face")
    else:
        dia_options = st.multiselect(
            "auto bar diameters",
            REBAR_DIAMETERS_MM,
            default=[20.0, 25.0, 28.0, 32.0],
            key="dia_options",
            format_func=lambda dia: f"{rebar_label(dia)}  fy={rebar_fy_mpa(dia):.0f} MPa",
        )


st.subheader("Bearing Loads")
load_cols = st.columns([1, 1, 1, 1])
with load_cols[0]:
    bearing_rows = st.radio("Bearing rows", [1, 2], horizontal=True, key="bearing_rows")
with load_cols[1]:
    bearings_per_row = st.number_input("Bearings per row", min_value=1, max_value=20, value=4, step=1, key="bearings_per_row")
with load_cols[2]:
    if "row_spacing_y_mm" in st.session_state:
        st.session_state.row_spacing_y_mm = min(float(st.session_state.row_spacing_y_mm), float(depth_y_mm))
    row_spacing_y_mm = st.number_input(
        "Row spacing y (mm)",
        min_value=0.0,
        max_value=float(depth_y_mm),
        value=min(600.0, float(depth_y_mm)),
        step=50.0,
        disabled=int(bearing_rows) == 1,
        key="row_spacing_y_mm",
    )
with load_cols[3]:
    reset_table = st.button("Reset layout", width="stretch", key="reset_layout_button")

expected_bearing_count = int(bearing_rows) * int(bearings_per_row)

if "bearing_table" not in st.session_state or reset_table:
    default_table = default_bearings(
        int(bearings_per_row),
        int(bearing_rows),
        width_x_mm,
        height_z_mm,
        row_spacing_y_mm,
    )
    st.session_state.bearing_table = default_table
    st.session_state.pending_bearing_table = default_table.copy()
    st.session_state.bearing_table_dirty = False
    st.session_state.pop("bearing_load_editor", None)
elif len(st.session_state.bearing_table) != expected_bearing_count:
    default_table = default_bearings(
        int(bearings_per_row),
        int(bearing_rows),
        width_x_mm,
        height_z_mm,
        row_spacing_y_mm,
    )
    st.session_state.bearing_table = default_table
    st.session_state.pending_bearing_table = default_table.copy()
    st.session_state.bearing_table_dirty = False
    st.session_state.pop("bearing_load_editor", None)
elif "pending_bearing_table" not in st.session_state or len(st.session_state.pending_bearing_table) != expected_bearing_count:
    st.session_state.pending_bearing_table = clean_bearings(st.session_state.bearing_table).copy()
    st.session_state.bearing_table_dirty = False

editor_key = "bearing_load_editor"
bearing_update_message = st.session_state.pop("bearing_update_message", None)
if bearing_update_message:
    st.success(bearing_update_message)

with st.form("bearing_load_form"):
    edited_bearing_table = st.data_editor(
        st.session_state.pending_bearing_table,
        key=editor_key,
        num_rows="fixed",
        width="stretch",
        hide_index=True,
        column_config={
            "name": st.column_config.TextColumn("Bearing"),
            "x_mm": st.column_config.NumberColumn("x (mm)", step=50.0, format="%.0f"),
            "y_mm": st.column_config.NumberColumn("y (mm)", step=50.0, format="%.0f"),
            "z_mm": st.column_config.NumberColumn("z (mm)", step=50.0, format="%.0f"),
            "Pu_x_kN": st.column_config.NumberColumn("Pu_x (kN)", step=10.0, format="%.1f"),
            "Pu_y_kN": st.column_config.NumberColumn("Pu_y (kN)", step=10.0, format="%.1f"),
            "Pu_z_kN": st.column_config.NumberColumn("Pu_z comp. (kN)", step=10.0, format="%.1f"),
            "Mu_x_kNm": st.column_config.NumberColumn("Mu_x (kN-m)", step=10.0, format="%.1f"),
            "Mu_y_kNm": st.column_config.NumberColumn("Mu_y (kN-m)", step=10.0, format="%.1f"),
        },
    )
    update_data = st.form_submit_button("Update Data", type="primary", width="stretch")
if update_data:
    committed_bearing_table = committed_editor_table(
        edited_bearing_table,
        st.session_state.pending_bearing_table,
        editor_key,
    )
    st.session_state.bearing_table = committed_bearing_table
    st.session_state.pending_bearing_table = committed_bearing_table.copy()
    st.session_state.bearing_table_dirty = False
    st.session_state.bearing_update_message = "Bearing load data updated. Calculations now use the latest table."
    st.rerun()

bearings_df = clean_bearings(st.session_state.bearing_table)
records = bearings_df.to_dict("records")
global_resultant = combine_bearing_loads(records)
dead_load_summary = abutment_dead_load_summary(
    width_x_mm=width_x_mm,
    depth_y_mm=depth_y_mm,
    height_z_mm=height_z_mm,
    unit_weight_kn_m3=concrete_unit_weight_kn_m3,
)
backfill_vertical_result = (
    backfill_vertical_load_summary(
        width_x_mm=width_x_mm,
        depth_y_mm=depth_y_mm,
        backfill_width_m=backfill_ev_width_m,
        backfill_height_m=backfill_ev_height_m,
        gamma_soil_kn_m3=earth_gamma_soil_kn_m3,
        factor=backfill_ev_factor,
        earth_pressure_direction=earth_pressure_direction,
    )
    if include_backfill_ev
    else None
)
approach_slab_result = (
    approach_slab_reaction_summary(
        slab_length_y_m=approach_slab_length_y_m,
        slab_width_x_m=approach_slab_width_x_m,
        slab_thickness_m=approach_slab_thickness_m,
        seat_length_m=approach_slab_seat_length_m,
        concrete_unit_weight_kn_m3=concrete_unit_weight_kn_m3,
        superimposed_dead_kpa=approach_slab_superimposed_dead_kpa,
        gap_m=approach_slab_gap_m,
        live_load_preset=approach_slab_ll_preset,
        live_q_kpa=approach_slab_ll_q_kpa,
        live_loaded_width_m=approach_slab_ll_loaded_width_m,
        live_line_load_kn_m=approach_slab_ll_line_load_kn_m,
        track_count=int(approach_slab_track_count),
        point_load_kn=approach_slab_point_load_kn,
        point_load_y_m=approach_slab_point_load_y_m,
        dc_factor=approach_slab_dc_factor,
        dw_factor=approach_slab_dw_factor,
        ll_factor=approach_slab_ll_factor,
        depth_y_mm=depth_y_mm,
        earth_pressure_direction=earth_pressure_direction,
    )
    if include_approach_slab_reaction
    else None
)
earth_pressure_summaries: list[EarthPressureSummary] = []
if include_earth_pressure:
    earth_pressure_summaries = [
        earth_pressure_summary(
            width_x_mm=width_x_mm,
            height_m=earth_backfill_height_m,
            gamma_soil_kn_m3=earth_gamma_soil_kn_m3,
            phi_deg=earth_phi_deg,
            pressure_mode=earth_pressure_mode,
            q_other_kpa=q_other_kpa,
            live_q_kpa=traffic_q_kpa,
            live_h_eq_m=traffic_h_eq_m,
            direction=earth_pressure_direction,
            factors=factors,
        )
        for factors in earth_pressure_factor_candidates
    ]
earth_pressure_result = governing_earth_pressure_summary(earth_pressure_summaries)
stem_bar_fy_mpa = rebar_fy_mpa(float(stem_bar_dia_mm)) if earth_pressure_result is not None else 390.0
stem_min_rho, stem_min_basis = shrinkage_temperature_ratio(code_choice, stem_bar_fy_mpa)
stem_design_result = (
    stem_strip_design_summary(
        earth_pressure=earth_pressure_result,
        depth_y_mm=depth_y_mm,
        cover_mm=cover_mm,
        bar_dia_mm=float(stem_bar_dia_mm),
        fc_mpa=fc_mpa,
        fy_mpa=stem_bar_fy_mpa,
        phi_flexure=phi_flexure,
        min_rho=stem_min_rho,
    )
    if earth_pressure_result is not None
    else None
)

st.subheader("Strength Design Strip")
bearing_names = [str(row.get("name", "")) for row in records]
default_strength_names = default_strength_bearing_names(records)
stored_strength_names = st.session_state.get("strength_bearing_names", default_strength_names)
stored_strength_names = [name for name in stored_strength_names if name in bearing_names]
if not stored_strength_names:
    stored_strength_names = default_strength_names
st.session_state.strength_bearing_names = stored_strength_names

strip_cols = st.columns([1.1, 1.4, 1.1])
with strip_cols[0]:
    strip_width_mode = st.radio(
        "Strength section width",
        ["Auto effective strip", "Manual strip width", "Full abutment width"],
        horizontal=False,
        key="strip_width_mode",
        help="Use an effective strip for local bearing checks. Full width is mainly for global wall-line checks.",
    )
with strip_cols[1]:
    selected_strength_names = st.multiselect(
        "Bearings included in strength strip",
        options=bearing_names,
        key="strength_bearing_names",
        disabled=strip_width_mode == "Full abutment width",
        help="Default selects the critical x-line of bearings. For a pier with two rows, bearings with the same x are selected together.",
    )

selected_strength_records = selected_bearings(records, selected_strength_names)
if strip_width_mode != "Full abutment width" and not selected_strength_records:
    selected_strength_names = default_strength_names
    selected_strength_records = selected_bearings(records, selected_strength_names)

strip_info = effective_strip_recommendation(
    records=records,
    selected_names=selected_strength_names,
    full_width_x_mm=width_x_mm,
    depth_y_mm=depth_y_mm,
    bearing_size_x_mm=bearing_size_x_mm,
)
auto_strip_width_x_mm = strip_info["recommended_width_mm"]
if strip_width_mode == "Full abutment width":
    strength_design_width_x_mm = width_x_mm
    strength_strip_center_x_mm = 0.0
    strength_records_global = records
    strength_display_names = bearing_names
elif strip_width_mode == "Manual strip width":
    with strip_cols[2]:
        manual_strip_min = max(100.0, bearing_size_x_mm)
        manual_strip_default = float(min(width_x_mm, max(auto_strip_width_x_mm, bearing_size_x_mm)))
        if st.session_state.get("manual_strip_width_mm") is None:
            st.session_state.pop("manual_strip_width_mm", None)
        elif "manual_strip_width_mm" in st.session_state:
            st.session_state.manual_strip_width_mm = min(
                float(width_x_mm),
                max(manual_strip_min, float(st.session_state.manual_strip_width_mm)),
            )
        strength_design_width_x_mm = st.number_input(
            "manual strip width (mm)",
            min_value=manual_strip_min,
            max_value=float(width_x_mm),
            value=manual_strip_default,
            step=50.0,
            key="manual_strip_width_mm",
            help="Use this when the project specification defines another effective width.",
        )
    strength_strip_center_x_mm = strip_center_x_mm(selected_strength_records)
    strength_records_global = selected_strength_records
    strength_display_names = selected_strength_names
else:
    with strip_cols[2]:
        st.metric("recommended beff", f"{auto_strip_width_x_mm:,.0f} mm")
    strength_design_width_x_mm = auto_strip_width_x_mm
    strength_strip_center_x_mm = strip_center_x_mm(selected_strength_records)
    strength_records_global = selected_strength_records
    strength_display_names = selected_strength_names

strength_records = localize_bearing_records(strength_records_global, strength_strip_center_x_mm)
bearing_resultant = combine_bearing_loads(strength_records)
resultant, earth_design_width_ratio, earth_design_vy_kn, earth_design_mux_knm = section_design_resultant_with_earth_pressure(
    bearing_resultant,
    earth_pressure_result,
    strength_design_width_x_mm,
)
spacing_text = (
    f"{strip_info['spacing_limit_mm']:,.0f} mm"
    if math.isfinite(strip_info["spacing_limit_mm"])
    else "not limited"
)
edge_text = (
    f"left {strip_info['left_edge_center_distance_mm']:,.0f} mm, "
    f"right {strip_info['right_edge_center_distance_mm']:,.0f} mm"
)
st.caption(
    "Strength check uses local x about the selected strip center. "
    f"Effective width guide: loaded width + 4t = {strip_info['distribution_limit_mm']:,.0f} mm, "
    f"tributary/edge limit = {strip_info['tributary_limit_mm']:,.0f} mm "
    f"(edge-center distances: {edge_text}; typical c/c spacing = {spacing_text}), "
    f"available length = {strip_info['available_limit_mm']:,.0f} mm."
)
if strip_width_mode != "Full abutment width":
    st.info(
        f"Design strip: {', '.join(strength_display_names)} | "
        f"center x = {strength_strip_center_x_mm:,.0f} mm | "
        f"section used for strength = {strength_design_width_x_mm:,.0f} x {depth_y_mm:,.0f} mm"
    )
if earth_pressure_result is not None:
    st.info(
        "Section design includes earth pressure over the selected strip: "
        f"Vy = {earth_design_vy_kn:,.2f} kN, "
        f"Mux = {earth_design_mux_knm:,.2f} kN-m "
        f"({earth_design_width_ratio:.1%} of full-width earth pressure)."
    )

if resultant.pu_kn < 0:
    st.warning("Pu_z resultant is net tension. The app can show resultants, but reinforcement design assumptions should be checked carefully.")

check = None
design_error = None
bearing_data_pending = st.session_state.get("bearing_table_dirty", False)
if bearing_data_pending:
    design_error = "Bearing load edits are pending. Click Update Data to run calculations with the latest table."
else:
    try:
        if mode == "Auto design":
            if not dia_options:
                design_error = "Select at least one bar diameter for auto design."
            else:
                with st.spinner("Searching reinforcement layout..."):
                    check = find_reinforcement(
                        width_x_mm=strength_design_width_x_mm,
                        depth_y_mm=depth_y_mm,
                        cover_mm=cover_mm,
                        bar_dia_options_mm=tuple(dia_options),
                        fc_mpa=fc_mpa,
                        es_mpa=es_mpa,
                        pu_kn=resultant.pu_kn,
                        mux_knm=resultant.mux_knm,
                        muy_knm=resultant.muy_knm,
                        params=params,
                        rho_min_percent=rho_min_percent,
                        rho_max_percent=rho_max_percent,
                        biaxial_method=biaxial_method,
                        load_contour_alpha=load_contour_alpha,
                        min_clear_spacing_mm=min_clear_spacing_mm,
                        max_spacing_advisory_mm=max_spacing_advisory_mm,
                        enforce_max_spacing=enforce_max_spacing,
                    )
                if check is None:
                    design_error = "No reinforcement layout passed within the selected auto limits."
        else:
            check = analyze_section(
                width_x_mm=strength_design_width_x_mm,
                depth_y_mm=depth_y_mm,
                cover_mm=cover_mm,
                bar_dia_mm=float(bar_dia_mm),
                bars_x_face=int(bars_x_face),
                bars_y_face=int(bars_y_face),
                fc_mpa=fc_mpa,
                fy_mpa=selected_fy_mpa,
                es_mpa=es_mpa,
                pu_kn=resultant.pu_kn,
                mux_knm=resultant.mux_knm,
                muy_knm=resultant.muy_knm,
                params=params,
                biaxial_method=biaxial_method,
                load_contour_alpha=load_contour_alpha,
                max_spacing_advisory_mm=max_spacing_advisory_mm,
            )
    except Exception as exc:  # noqa: BLE001
        design_error = str(exc)


tabs = st.tabs(["Views", "Results", "Section", "Shear", "Method"])

with tabs[0]:
    view_cols = st.columns(2)
    with view_cols[0]:
        st.plotly_chart(
            plan_view(
                records,
                width_x_mm=width_x_mm,
                depth_y_mm=depth_y_mm,
                pilecap_overhang_mm=pilecap_overhang_mm,
                bearing_size_x_mm=bearing_size_x_mm,
                bearing_size_y_mm=bearing_size_y_mm,
                selected_bearing_names=strength_display_names,
                strip_center_x_mm=strength_strip_center_x_mm,
                strip_width_x_mm=strength_design_width_x_mm,
            ),
            width="stretch",
        )
    with view_cols[1]:
        st.plotly_chart(
            front_view(
                records,
                width_x_mm=width_x_mm,
                height_z_mm=height_z_mm,
                pilecap_overhang_mm=pilecap_overhang_mm,
                pilecap_thickness_mm=pilecap_thickness_mm,
                bearing_size_x_mm=bearing_size_x_mm,
                bearing_size_y_mm=bearing_size_y_mm,
            ),
            width="stretch",
        )
    st.plotly_chart(
        side_view(
            records,
            depth_y_mm=depth_y_mm,
            height_z_mm=height_z_mm,
            pilecap_overhang_mm=pilecap_overhang_mm,
            pilecap_thickness_mm=pilecap_thickness_mm,
            bearing_size_x_mm=bearing_size_x_mm,
            bearing_size_y_mm=bearing_size_y_mm,
            earth_pressure=earth_pressure_result,
            approach_slab=approach_slab_result,
        ),
        width="stretch",
    )

with tabs[1]:
    metric_row(resultant, check, strength_design_width_x_mm)
    st.markdown(
        '<p class="small-note">Section design uses the displayed Pu, Mux, and Muy. Vx and Vy are checked separately in the Shear tab as preliminary sectional shear.</p>',
        unsafe_allow_html=True,
    )
    with st.expander("Section design demand breakdown", expanded=earth_pressure_result is not None):
        demand_rows = [
            ["Bearing strip Pu", bearing_resultant.pu_kn, "kN"],
            ["Bearing strip Vy", bearing_resultant.vy_kn, "kN"],
            ["Earth pressure Vy added to strip", earth_design_vy_kn, "kN"],
            ["Section design Vy", resultant.vy_kn, "kN"],
            ["Bearing strip Mux", bearing_resultant.mux_knm, "kN-m"],
            ["Earth pressure Mux added to strip", earth_design_mux_knm, "kN-m"],
            ["Section design Mux", resultant.mux_knm, "kN-m"],
            ["Section design Muy", resultant.muy_knm, "kN-m"],
        ]
        st.dataframe(pd.DataFrame(demand_rows, columns=["Demand item", "Value", "Unit"]), width="stretch", hide_index=True)
        if earth_pressure_result is not None:
            st.caption(
                "Earth pressure is scaled by the selected strength-section width before being added to Mux. "
                f"Scale = {earth_design_width_ratio:.1%} of the full abutment width."
            )
    with st.expander("Global all-bearing resultants", expanded=False):
        global_table = pd.DataFrame(
            [
                ["Pu compression", global_resultant.pu_kn, "kN"],
                ["Vx resultant", global_resultant.vx_kn, "kN"],
                ["Vy resultant", global_resultant.vy_kn, "kN"],
                ["Mux about global origin", global_resultant.mux_knm, "kN-m"],
                ["Muy about global origin", global_resultant.muy_knm, "kN-m"],
                ["Tz resultant", global_resultant.torsion_z_knm, "kN-m"],
            ],
            columns=["Resultant", "Value", "Unit"],
        )
        st.dataframe(global_table, width="stretch", hide_index=True)
    if design_error:
        st.error(design_error)
    elif check is not None:
        st.markdown(status_html(check.status, check.governing_ratio), unsafe_allow_html=True)
        st.progress(min(1.0, max(0.0, check.governing_ratio)))
        summary = pd.DataFrame(
            [row for row in [
                ["Axial Pu / phi Pmax", check.axial_ratio],
                ["Mux / phi Mnx(Pu)", check.mux_ratio],
                ["Muy / phi Mny(Pu)", check.muy_ratio],
                ["Linear biaxial interaction", check.biaxial_linear_ratio],
                [f"Load contour interaction, alpha={check.load_contour_alpha:.2f}", check.load_contour_ratio],
                ["PMM surface radial interaction", check.pmm_ratio],
                [f"Selected biaxial method: {check.biaxial_method}", check.biaxial_ratio],
                ["Governing utilization", check.governing_ratio],
            ] if not math.isnan(float(row[1]))],
            columns=["Check", "Ratio"],
        )
        st.dataframe(summary, width="stretch", hide_index=True)

        rebar_text = (
            f"{check.bar_count} bars {rebar_label(check.bar_dia_mm)} fy={check.fy_mpa:.0f} MPa: "
            f"{check.bars_x_face} bars along x on each top/bottom face, "
            f"{check.bars_y_face} bars along y on each left/right face"
        )
        st.info(rebar_text)
        min_reinf = minimum_reinforcement_check(check, code_choice)
        st.subheader("Minimum Reinforcement Check")
        st.caption(
            "Shrinkage/temperature-style distributed longitudinal reinforcement only. "
            "Column longitudinal minimum such as 1%Ag is intentionally not applied for this abutment/pier strip check."
        )
        min_reinf_table = pd.DataFrame(
            [
                ["basis", min_reinf["basis"], "", ""],
                ["required rho", f"{float(min_reinf['rho_req_percent']):.3f}", "%", ""],
                ["As,min total", f"{float(min_reinf['as_min_total_mm2']):,.0f}", "mm2", ""],
                ["As provided total", f"{float(min_reinf['as_provided_total_mm2']):,.0f}", "mm2", str(min_reinf["total_status"])],
                ["As,min each main face", f"{float(min_reinf['as_min_each_main_face_mm2']):,.0f}", "mm2", ""],
                [
                    "As provided each top/bottom face",
                    f"{float(min_reinf['as_provided_each_top_bottom_face_mm2']):,.0f}",
                    "mm2",
                    str(min_reinf["face_status"]),
                ],
                ["max c/c spacing advisory", f"{check.max_center_spacing_mm:,.0f} / {check.max_spacing_advisory_mm:,.0f}", "mm", check.spacing_status],
            ],
            columns=["Item", "Value", "Unit", "Status"],
        )
        st.dataframe(min_reinf_table, width="stretch", hide_index=True)
        if min_reinf["total_status"] != "OK" or min_reinf["face_status"] != "OK":
            st.warning(
                "Shrinkage/temperature minimum reinforcement is not satisfied. Increase bar count, bar size, "
                "or adjust the effective strip dimensions used for the check."
            )
        if check.min_clear_spacing_mm < min_clear_spacing_mm:
            st.warning(
                f"Clear spacing is {check.min_clear_spacing_mm:.0f} mm, less than the selected minimum "
                f"{min_clear_spacing_mm:.0f} mm. Adjust bar count, bar size, cover, or section dimensions."
            )
        if check.spacing_status != "OK":
            st.warning(
                f"Maximum center-to-center spacing is {check.max_center_spacing_mm:.0f} mm, greater than the "
                f"advisory limit {check.max_spacing_advisory_mm:.0f} mm."
            )

    st.subheader("Pile Cap Base Force Summary")
    pilecap_load_notes = []
    if earth_pressure_result is not None:
        pilecap_load_notes.append("selected lateral earth pressure")
    if backfill_vertical_result is not None:
        pilecap_load_notes.append("vertical backfill EV")
    if approach_slab_result is not None:
        pilecap_load_notes.append("approach slab reaction")
    earth_caption = f"and {' plus '.join(pilecap_load_notes)}" if pilecap_load_notes else "with earth/backfill loads excluded"
    st.caption(
        f"Uses all bearing loads transferred to the abutment/pier base centroid, abutment/pier self-weight, "
        f"{earth_caption}. "
        f"Coordinate origin is the centroid used by the bearing table; x width = {width_x_mm / 1000.0:,.3f} m. "
        "Compression Pu_z is positive, and moment signs follow the displayed axes."
    )
    dl_cols = st.columns(4)
    dl_cols[0].metric("DL volume", f"{dead_load_summary.volume_m3:,.3f} m3")
    dl_cols[1].metric("Service DL", f"{dead_load_summary.service_dl_kn:,.2f} kN")
    dl_cols[2].metric("DL factor", f"{dead_load_summary.factor:.2f}")
    dl_cols[3].metric("Pu_z from 1.40D", f"{dead_load_summary.uls_pu_z_kn:,.2f} kN")
    if backfill_vertical_result is not None:
        ev_cols = st.columns(4)
        ev_cols[0].metric("EV volume", f"{backfill_vertical_result.volume_m3:,.3f} m3")
        ev_cols[1].metric("Service EV", f"{backfill_vertical_result.service_ev_kn:,.2f} kN")
        ev_cols[2].metric("EV factor", f"{backfill_vertical_result.factor:.2f}")
        ev_cols[3].metric("Pu_z from EV", f"{backfill_vertical_result.uls_pu_z_kn:,.2f} kN")
    if approach_slab_result is not None:
        as_cols = st.columns(4)
        as_cols[0].metric("AS l_gap", f"{approach_slab_result.gap_m:,.2f} m")
        as_cols[1].metric("AS service DC+DW", f"{approach_slab_result.service_dc_kn + approach_slab_result.service_dw_kn:,.2f} kN")
        as_cols[2].metric("AS service LL", f"{approach_slab_result.service_ll_kn:,.2f} kN")
        as_cols[3].metric("Pu_z from AS", f"{approach_slab_result.uls_pu_z_kn:,.2f} kN")
    if earth_pressure_result is not None:
        ep_cols = st.columns(4)
        ep_cols[0].metric("Earth K", f"{earth_pressure_result.pressure_coefficient:.3f}", delta=earth_pressure_result.combination, delta_color="off")
        ep_cols[1].metric("EH service", f"{earth_pressure_result.service_soil_kn_per_m:,.2f} kN/m")
        ep_cols[2].metric(
            "LS surcharge",
            f"{earth_pressure_result.service_live_kn_per_m:,.2f} kN/m",
            delta=f"h_eq {earth_pressure_result.live_h_eq_m:.2f} m",
            delta_color="off",
        )
        ep_cols[3].metric("ULS Vy earth", f"{earth_pressure_result.uls_vy_kn:,.2f} kN")
        with st.expander("Earth pressure ULS factor candidates", expanded=earth_pressure_result.load_code.startswith("EN")):
            st.dataframe(earth_pressure_factor_table(earth_pressure_summaries), width="stretch", hide_index=True)
            st.caption("For EN auto governing, the app uses the row with the largest absolute earth-pressure Mu_x.")
    elif structure_type == "Wall Pier / Pier wall without backfill":
        st.info("Wall Pier mode: earth pressure is excluded from the pile cap summary and from abutment stem design.")
    st.dataframe(
        pilecap_base_force_summary_table(
            global_resultant,
            width_x_mm,
            dead_load_summary,
            backfill_vertical_result,
            approach_slab_result,
            earth_pressure_result,
        ),
        width="stretch",
        hide_index=True,
    )
    if stem_design_result is not None:
        st.subheader("Abutment Stem Design - 1 m Strip")
        stem_table = pd.DataFrame(
            [
                ["Governing earth pressure combination", f"{earth_pressure_result.load_code} - {earth_pressure_result.combination}", "", ""],
                ["Mu demand at stem base", f"{stem_design_result.mu_kNm_per_m:,.2f}", "kN-m/m", ""],
                ["Vu demand at stem base", f"{stem_design_result.vu_kN_per_m:,.2f}", "kN/m", ""],
                ["Effective depth d", f"{stem_design_result.effective_depth_mm:,.0f}", "mm", ""],
                ["As required by flexure", f"{stem_design_result.as_flexure_mm2_per_m:,.0f}", "mm2/m", ""],
                ["As minimum guide", f"{stem_design_result.as_min_mm2_per_m:,.0f}", "mm2/m", stem_min_basis],
                ["As governing", f"{stem_design_result.as_required_mm2_per_m:,.0f}", "mm2/m", stem_design_result.status],
                [
                    f"Spacing guide for {rebar_label(stem_design_result.bar_dia_mm)}",
                    f"{stem_design_result.bar_spacing_mm:,.0f}" if math.isfinite(stem_design_result.bar_spacing_mm) else "N/A",
                    "mm",
                    "provide <= this spacing",
                ],
            ],
            columns=["Item", "Value", "Unit", "Status"],
        )
        st.dataframe(stem_table, width="stretch", hide_index=True)
        st.caption(
            "This is a preliminary vertical-flexure guide for a 1 m abutment stem strip under earth pressure only. "
            "It is separate from the bearing PMM strip check and should be verified with the final code/owner detailing rules."
        )
    pilecap_note = (
        "Self-weight is calculated as a rectangular abutment/pier block from the current Geometry inputs "
        "(Lx x t x H x concrete unit weight). It is applied at the centroid, so it adds Pu_z only and no moment. "
    )
    if earth_pressure_result is not None:
        pilecap_note += (
            "Earth pressure is applied along the selected y direction; triangular soil pressure acts at H/3 above the base, "
            "while q_other and traffic live load surcharge act at H/2. "
        )
    else:
        pilecap_note += "Earth pressure is excluded for the selected structure mode/settings. "
    if backfill_vertical_result is not None:
        pilecap_note += (
            "Backfill EV is calculated from gamma_soil x Lx x b_EV x h_EV and placed at the centroid of the supported "
            "backfill width on the heel/backfill side, so it adds Pu_z and an eccentric Mu_x to the pile-cap summary. "
        )
    else:
        pilecap_note += "Vertical backfill EV is excluded for the selected settings. "
    if approach_slab_result is not None:
        pilecap_note += (
            "Approach slab reaction uses the selected soil-supported l_gap, so only loads over the assumed gap bridge "
            "to the abutment; soil-supported slab length is not added as abutment reaction. "
        )
        if approach_slab_ll_replaces_surcharge:
            pilecap_note += "Traffic q_LS surcharge is suppressed to avoid double counting the approach slab live load. "
    else:
        pilecap_note += "Approach slab reaction is excluded for the selected settings. "
    pilecap_note += (
        "Mu_x can be linearized as a distributed line couple along x with unit kN-m/m when the pile-cap model "
        "accepts wall-line moment input. It is an idealized smear of the total couple, not a vertical line load. "
        "Mu_y is kept as a point couple at the centroid because smearing it uniformly along x would hide the "
        "longitudinal eccentricity that creates bending about y."
    )
    st.info(pilecap_note)

with tabs[2]:
    if check is None:
        st.info("Section check will appear after a valid reinforcement layout is available.")
    else:
        sec_cols = st.columns(2)
        with sec_cols[0]:
            st.plotly_chart(reinforcement_plan(check), width="stretch")
            st.markdown(minimum_reinforcement_box_html(check, code_choice), unsafe_allow_html=True)
        with sec_cols[1]:
            st.plotly_chart(
                interaction_plot(check, resultant.pu_kn, resultant.design_mux_knm, resultant.design_muy_knm),
                width="stretch",
            )
        if check.biaxial_method == PMM_METHOD:
            st.plotly_chart(
                pmm_slice_plot(check, resultant.mux_knm, resultant.muy_knm, resultant.pu_kn),
                width="stretch",
            )
            st.plotly_chart(
                pmm_surface_plot(check, resultant.mux_knm, resultant.muy_knm, resultant.pu_kn),
                width="stretch",
            )
        else:
            st.plotly_chart(
                biaxial_load_contour_plot(check, resultant.mux_knm, resultant.muy_knm),
                width="stretch",
            )
        bar_table = pd.DataFrame(
            [{"bar": idx + 1, "x_mm": bar.x_mm, "y_mm": bar.y_mm, "area_mm2": bar.area_mm2} for idx, bar in enumerate(check.bars)]
        )
        st.dataframe(bar_table, width="stretch", hide_index=True)

with tabs[3]:
    st.subheader("Shear Check")
    st.caption(
        "Preliminary sectional shear check using the same code assumption selected in Design Basis. "
        "Pile-cap one-way/punching shear and detailed AASHTO theta/beta MCFT design remain separate final-design checks."
    )
    shear_cols = st.columns(3)
    with shear_cols[0]:
        shear_tie_dia_mm = st.selectbox(
            "Shear tie / stirrup diameter",
            SHEAR_REBAR_DIAMETERS_MM,
            index=SHEAR_REBAR_DIAMETERS_MM.index(12.0),
            key="shear_tie_dia_mm",
            format_func=lambda dia: f"{rebar_label(dia)} fy={rebar_fy_mpa(dia):.0f} MPa",
        )
    with shear_cols[1]:
        shear_tie_legs = st.number_input(
            "Number of shear legs",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            key="shear_tie_legs",
            help="Total legs crossing the shear crack per spacing line.",
        )
    with shear_cols[2]:
        shear_spacing_provided_mm = st.number_input(
            "Provided shear spacing s (mm)",
            min_value=25.0,
            max_value=1000.0,
            value=200.0,
            step=25.0,
            key="shear_spacing_provided_mm",
        )

    shear_longitudinal_bar_dia = check.bar_dia_mm if check is not None else float(st.session_state.get("bar_dia_mm", 25.0))
    shear_fyt_mpa = rebar_fy_mpa(float(shear_tie_dia_mm))
    shear_d_y_mm = depth_y_mm - cover_mm - shear_longitudinal_bar_dia / 2.0
    shear_d_x_mm = strength_design_width_x_mm - cover_mm - shear_longitudinal_bar_dia / 2.0
    shear_checks = [
        shear_check_summary(
            name="Strength section shear Vy",
            axis="Vy / Mux direction",
            vu_kn=resultant.vy_kn,
            bw_mm=strength_design_width_x_mm,
            d_mm=shear_d_y_mm,
            fc_mpa=fc_mpa,
            fyt_mpa=shear_fyt_mpa,
            code_name=code_choice,
            tie_dia_mm=float(shear_tie_dia_mm),
            tie_legs=int(shear_tie_legs),
            spacing_provided_mm=shear_spacing_provided_mm,
        ),
        shear_check_summary(
            name="Strength section shear Vx",
            axis="Vx / Muy direction",
            vu_kn=resultant.vx_kn,
            bw_mm=depth_y_mm,
            d_mm=shear_d_x_mm,
            fc_mpa=fc_mpa,
            fyt_mpa=shear_fyt_mpa,
            code_name=code_choice,
            tie_dia_mm=float(shear_tie_dia_mm),
            tie_legs=int(shear_tie_legs),
            spacing_provided_mm=shear_spacing_provided_mm,
        ),
    ]
    if stem_design_result is not None:
        shear_checks.append(
            shear_check_summary(
                name="Abutment stem 1 m strip shear",
                axis="Earth-pressure Vy per m",
                vu_kn=stem_design_result.vu_kN_per_m,
                bw_mm=1000.0,
                d_mm=stem_design_result.effective_depth_mm,
                fc_mpa=fc_mpa,
                fyt_mpa=shear_fyt_mpa,
                code_name=code_choice,
                tie_dia_mm=float(shear_tie_dia_mm),
                tie_legs=int(shear_tie_legs),
                spacing_provided_mm=shear_spacing_provided_mm,
            )
        )

    shear_table = pd.DataFrame(
        [
            {
                "Check": item.name,
                "Axis": item.axis,
                "Status": item.status,
                "Vu": f"{item.vu_kn:,.2f} kN",
                "phi Vc": f"{item.phi_vc_kn:,.2f} kN",
                "Vs req": f"{item.vs_required_kn:,.2f} kN",
                "Av/s req": f"{item.avs_required_mm2_per_m:,.0f} mm2/m",
                "s req": "N/A" if not math.isfinite(item.spacing_required_mm) else f"{item.spacing_required_mm:,.0f} mm",
                "s max": f"{item.spacing_code_max_mm:,.0f} mm",
                "s recommended": "N/A" if not math.isfinite(item.spacing_recommended_mm) else f"{item.spacing_recommended_mm:,.0f} mm",
                "s provided": f"{item.spacing_provided_mm:,.0f} mm",
                "phi Vn provided": f"{item.phi_vn_provided_kn:,.2f} kN",
                "U": f"{item.utilization:.3f}",
            }
            for item in shear_checks
        ]
    )
    st.dataframe(shear_table, width="stretch", hide_index=True)
    if any(item.status != "OK" for item in shear_checks):
        st.warning("At least one shear check is NG. Reduce spacing, increase shear legs/bar size, or increase section dimensions.")

    with st.expander("Shear calculation details", expanded=False):
        detail_rows = []
        for item in shear_checks:
            detail_rows.extend(
                [
                    [item.name, "code basis", item.code_basis, ""],
                    [item.name, "section bw x d", f"{item.bw_mm:,.0f} x {item.d_mm:,.0f}", "mm"],
                    [item.name, "phi_v", f"{item.phi_v:.2f}", ""],
                    [item.name, "Vc nominal", f"{item.vc_kn:,.2f}", "kN"],
                    [item.name, "Av/s strength", f"{item.avs_strength_mm2_per_m:,.0f}", "mm2/m"],
                    [item.name, "Av/s minimum", f"{item.avs_min_mm2_per_m:,.0f}", "mm2/m"],
                    [item.name, "provided shear steel", f"{item.tie_legs} legs {rebar_label(item.tie_dia_mm)} @ {item.spacing_provided_mm:,.0f}", "mm"],
                    [item.name, "Av/s provided", f"{item.avs_provided_mm2_per_m:,.0f}", "mm2/m"],
                    [item.name, "notes", item.notes, ""],
                ]
            )
        st.dataframe(pd.DataFrame(detail_rows, columns=["Check", "Item", "Value", "Unit"]), width="stretch", hide_index=True)
    st.info(
        "For wall/pier shear, Vy is checked with the selected strength width and wall thickness d. "
        "Vx is checked in the orthogonal direction. For abutments, an additional 1 m stem strip shear check is shown from earth pressure only."
    )


with tabs[4]:
    st.markdown(
        """
        **Coordinate and sign convention**

        x and y are plan axes. z is vertical upward. Bearing `Pu_z` is entered as positive downward compression.
        `Pu_x`, `Pu_y`, `Mu_x`, and `Mu_y` follow the displayed positive axes and the right-hand rule.
        Bearing layout can be generated as 1 row or 2 rows. For 2 rows, the generated y positions are `+s/2` and `-s/2`,
        where `s` is the row spacing. The generated table remains editable for custom pier layouts.

        **Fixed-base resultants**

        `Pu = sum(Pu_z)`  
        `Vx = sum(Pu_x)`, `Vy = sum(Pu_y)`  
        `Mux = sum(Mu_x + (-y Pu_z - z Pu_y) / 1000)`  
        `Muy = sum(Mu_y + (z Pu_x + x Pu_z) / 1000)`  
        `Tz = sum((x Pu_y - y Pu_x) / 1000)`

        **Structure type and generated load factoring**

        `Structure type = Abutment with backfill` enables earth pressure inputs, vertical backfill `EV`, section-design earth pressure, and the abutment stem strip design.
        `Structure type = Wall Pier / Pier wall without backfill` hides earth pressure and excludes it from all
        summaries. Bearing loads in the table are assumed to be already factored `ULS` loads from the user's chosen
        bridge load combination, so the app does not factor bearing loads again.

        The selected generated-earth-pressure code applies only to loads created inside the app: `EH`, `EV`,
        `q_other / ES`, and traffic live load surcharge `LS`.

        AASHTO LRFD Strength I factors used by the app:

        `EH active = 1.50`, `EH at-rest = 1.35`, `EV = 1.35`, `q_other / ES = 1.50`, `traffic LS = 1.75`

        EN / Eurocode DA1 auto governing runs two preliminary STR/GEO combinations and uses the row with the largest
        absolute earth-pressure `Mu_x`:

        `DA1 C1: A1 + M1`, with `gamma_G = 1.35`, traffic factor selected from road/rail/other, and `gamma_tan_phi = 1.00`  
        `DA1 C2: A2 + M2`, with `gamma_G = 1.00`, `gamma_Q = 1.30`, and `gamma_tan_phi = 1.25`

        Use `Custom factors` when the owner, geotechnical report, or National Annex specifies different values.

        **Strength section demand**

        The bearing table is not rewritten. For section design in Abutment mode, the app first calculates the selected
        bearing strip resultant, then adds the generated earth pressure over the selected strength-section width:

        `Mux_design = Mux_bearing_strip + Mux_earth x (beff / Lx)`  
        `Pu_design = Pu_bearing_strip`  
        `Muy_design = Muy_bearing_strip`

        `Vy` from earth pressure is reported in the demand breakdown, but the PMM strength check uses `Pu`, `Mux`,
        and `Muy`.

        **Pile cap base force summary**

        The pile cap summary in the Results tab uses all bearings, transfers their loads to the base centroid of
        the abutment/pier, adds the abutment/pier self-weight dead load, and includes lateral earth pressure plus
        vertical backfill `EV` only in
        Abutment mode. It is intended as an interface force summary for a separate pile-cap model, not as a pile-cap
        design check.

        `DL_self = gamma_c x Lx x t x H`  
        `Pu_z from DL = 1.40 x DL_self`  
        `EV_service = gamma_soil x Lx x b_EV x h_EV`  
        `Pu_z from EV = gamma_EV x EV_service`  
        `y_EV = backfill-side sign x (t / 2 + b_EV / 2)`  
        `Mu_x from EV = -y_EV x Pu_z from EV`  
        `Pu_z point = sum(Pu_z bearings) + Pu_z from DL + Pu_z from EV`  
        `Pu_z line = Pu_z point / Lx`, where `Lx` is the abutment/pier width along x  
        `Mu_x point = Mux` at the centroid  
        `Mu_x line couple = Mu_x point / Lx`, with unit `kN-m/m`  
        `Mu_y point = Muy` at the centroid

        The self-weight is assumed to act at the abutment/pier centroid, so it increases `Pu_z` only and does not
        add `Mu_x`, `Mu_y`, or `Tz`.

        **Approach slab reaction**

        Soil-supported approach slab mode uses an effective unsupported length `l_gap`. Only the slab and live load
        over this assumed settlement gap are treated as bridging to the abutment:

        `R_DC = 0.5 x gamma_c x t_slab x l_gap x B_slab`  
        `R_DW = 0.5 x q_DW x l_gap x B_slab`  
        `R_LL_area = 0.5 x q_LL x l_gap x loaded_width`  
        `R_LL_line = 0.5 x q_line x l_gap x number_of_tracks`  
        `R_point = P x (l_gap - y) / l_gap` for optional concentrated load inside the gap

        The approach slab reaction is placed at the backfill-side seat line of the abutment, so it adds `Pu_z` and
        eccentric `Mu_x` to the pile-cap summary. If the approach slab live load is included as vertical reaction,
        traffic live-load surcharge `q_LS` can be suppressed to avoid double counting the same live load.

        Approach slab reaction load factors can be selected separately as `AASHTO LRFD Strength I`, `EN / Eurocode
        DA1 C1`, `EN / Eurocode DA1 C2`, or `Custom factors`. This lets the pile-cap reaction summary follow the
        selected design basis instead of always using AASHTO-style factors.

        Earth pressure is calculated with the selected design `phi` value:

        `Ka = (1 - sin(phi)) / (1 + sin(phi))` for active pressure  
        `K0 = 1 - sin(phi)` for at-rest pressure  
        `P_EH = 0.5 x K x gamma_soil x H^2` per meter along x, acting at `H/3`  
        `P_q = K x q_other x H` per meter along x, acting at `H/2`  
        `q_LS = gamma_soil x h_eq` and `P_LS = K x q_LS x H` per meter along x, acting at `H/2`

        These earth pressure effects are added to the pile-cap interface `Vy` and `Mu_x`, and to the strength-section
        `Mu_x` demand over the selected strip width. They are not added back into the bearing table.

        **Abutment stem design**

        In Abutment mode, the Results tab also reports a preliminary 1 m strip stem design from earth pressure only:

        `Mu_stem = abs(Mu_x earth) / Lx`  
        `Vu_stem = abs(Vy earth) / Lx`

        The strip uses the wall thickness along y, the selected cover, the selected stem bar diameter, and the editable
        `phi flexure` value to estimate required vertical steel and a spacing guide. This stem strip design is separate
        from the PMM section check because it is a preliminary 1 m vertical-flexure design from earth pressure only,
        while the PMM section check combines the selected bearing strip with proportional earth-pressure `Mu_x`.

        **Shear check**

        The Shear tab checks the selected strength section in both plan directions:

        `Vy` uses the selected strength width as `bw` and the wall thickness direction as `d`.  
        `Vx` uses the wall thickness as `bw` and the selected strength width direction as `d`.

        For Abutment mode, a separate `1 m` stem strip shear check is also reported from earth-pressure `Vy / Lx`.
        ACI style uses a simplified `Vc = 0.17 sqrt(fc') bw d` with `phi_v = 0.75`. AASHTO style uses a simplified
        preliminary `phi_v = 0.90` shear check and should be verified with the project AASHTO LRFD theta/beta method
        for final design. Required shear reinforcement is reported as `Av/s`, with recommended spacing based on the
        selected tie diameter and number of legs.

        The `Mu_x / Lx` value is a distributed line couple along the wall length. It is theoretically usable when the
        pile-cap analysis model accepts a wall-line moment/couple input. It is not a vertical line load in `kN/m`.
        If the pile-cap model only accepts vertical loads, convert `Mu_x` into an equivalent compression/tension pair
        across the y-direction using the lever arm adopted in that model. `Mu_y` is kept as a point couple because it
        represents longitudinal eccentricity about the y-axis; smearing it uniformly along x can hide the bending
        distribution that the pile-cap model should resolve.

        **Effective design strip for Pn / PMM strength**

        The drawing and bearing table use the full abutment width. The strength check may use either the full width
        or a local effective strip. In effective strip mode, the app selects the governing bearing x-line by default,
        shifts the selected bearing coordinates to the strip center, then checks a rectangular section whose width is
        the selected strip width. This prevents the axial capacity `phi Pn` from being unconservatively inflated by
        using the full abutment length for a localized bearing reaction.

        The automatic strip recommendation is:

        `beff = min(l_loaded + 4t, b_trib, L_available)`

        where:

        `beff` = effective width used in the strength section  
`l_loaded` = loaded bearing/group width in x, taken as bearing size in x plus the distance between the outermost selected bearings  
        `t` = abutment thickness along y  
        `b_trib` = tributary/edge width in x. At an exterior bearing line, the edge side uses the distance
        from the abutment/pier free edge to the center of the outermost bearing; the interior side uses
        the midpoint to the adjacent bearing line. At an interior bearing line, both sides use midpoint
        tributary limits to adjacent bearing lines.  
        `L_available` = available abutment length between free edges, joints, or other physical limits

        The section properties for axial strength are then based on:

        `Ag = beff x t`

        and the reinforcement in the checked strip, rather than the full abutment plan length. Use the full abutment
        width only for a global wall-line check or when a refined load-distribution model justifies that the load is
        distributed over the full length.

        **Reference basis for effective width**

        ACI 318 wall provisions for concentrated vertical loads use the same load-distribution idea: unless a more
        detailed analysis demonstrates otherwise, the horizontal length of wall considered effective for each
        concentrated load is limited by the spacing to adjacent loads and by the loaded length plus four times the wall
        thickness. See ACI 318-14 Section 11.2.3.1; in later ACI editions, verify the corresponding wall/load
        distribution clause in the adopted code.

        ACI 318 Chapter 22 governs section strength for axial load with flexure, while Chapter 21 governs strength
        reduction factors. The effective strip width is therefore not a separate `phi Pn` equation; it defines the
        rectangular section dimensions used before calculating `phi Pn`, `phi Mnx`, `phi Mny`, and the PMM surface.

        AASHTO LRFD does not provide one universal effective abutment width for every bearing layout. Bridge practice
        commonly uses a design strip or per-unit-width substructure check unless a grillage/finite-element/refined
        distribution model is used. FHWA LRFD abutment design examples demonstrate this strip-style workflow for
        abutment stem/backwall design. Final width should follow the governing AASHTO edition, owner/authority design
        manual, joint layout, bearing spacing, diaphragm/load path, and any refined analysis used for the project.

        References:

        - ACI 318-14, Section 11.2.3.1: concentrated vertical load distribution in walls.
        - ACI 318-19, Chapters 21 and 22: strength reduction factors and sectional strength for axial load with flexure.
        - AASHTO LRFD Bridge Design Specifications: substructure analysis/design provisions and owner criteria for load distribution.
        - FHWA, LRFD Design Example: Abutment and Wingwall Design, illustrating strip/per-unit-width abutment design workflow.

        **RC section check**

        The app uses Whitney stress block strain compatibility for uniaxial `P-Mx` and `P-My` curves.
        `P-Mx` varies strain across the y direction and is mainly resisted by bars on the top/bottom faces.
        `P-My` varies strain across the x direction and is mainly resisted by bars on the left/right faces.
        Auto design searches by total steel area, rejects layouts below the selected clear spacing, then prefers layouts whose
        bar spacing is balanced around the section perimeter before checking strength.
        The max spacing advisory shown on the section is a detailing aid. Its default value is `min(3t, 450 mm)`,
        where `t` is the abutment thickness along y, but it must be verified against the governing ACI/AASHTO edition,
        member classification, seismic requirements, and project specifications.
        The Results tab also reports a shrinkage/temperature-style distributed longitudinal reinforcement check.
        The app intentionally does not apply a column longitudinal minimum such as `Ast >= 1%Ag` because the checked
        abutment/pier strip can have a very large gross area. For ACI style, the shrinkage/temperature ratio is taken
        as `rho = 0.0020` when `fy < 420 MPa`, otherwise `rho >= 0.0018*420/fy` but not less than `0.0014`.
        For AASHTO style, the app uses `As >= 0.11Ag/fy` with unit conversion for MPa. The check is reported for
        total vertical steel and for each main top/bottom face in the plan-section drawing.
        Biaxial bending can be checked with the conservative linear interaction:

        `|Mux| / phi Mnx(Pu) + |Muy| / phi Mny(Pu) <= 1.0`

        Or with the load contour method at the same axial load level:

        `(|Mux| / phi Mnx(Pu))^alpha + (|Muy| / phi Mny(Pu))^alpha <= 1.0`

        For `alpha = 1.00`, this reduces to the conservative linear moment interaction.
        For `alpha = 1.50`, the contour is rounded and typically less conservative; this value is a common
        Bresler/PCA-style approximation and should be accepted by the project reviewer before final use.
        ACI 318-19 Chapter 21 and Chapter 22 govern strength reduction factors and sectional strength for axial load
        with flexure. AASHTO LRFD Article 5.6.4.5 covers biaxial flexure checks; it does not make `alpha = 1.50`
        a universal requirement.

        The `PMM surface` method rotates the neutral axis through the section and calculates the design-strength
        surface `(phi Pn, phi Mnx, phi Mny)` from strain compatibility. The app then slices that 3D surface at
        the current `Pu` and checks the demand point by radial demand/capacity in the signed `Mux-Muy` plane.

        The biaxial contour graph is an `Mux-Muy` slice at the current `Pu`.
        The uniaxial P-M graph overlays two separate curves: `P-Mx` and `P-My`.
        Rebar yield strength is assigned by bar size: DB12, DB16, DB20, DB25, and DB28 use fy = 390 MPa;
        DB32 uses fy = 490 MPa. The current version assumes one vertical bar size for the checked section.

        ACI style uses strain-based phi interpolation. AASHTO LRFD style uses editable defaults with axial-to-flexural phi interpolation.
        The AASHTO default axial cap factor is `0.80`, consistent with tied compression members; edit this value only
        when another confinement condition or project criterion applies. In strain compatibility, bars inside the
        concrete compression stress block are counted with net steel force `As(fs - 0.85f'c)` so that the concrete
        displaced by compression reinforcement is not double-counted.
        Pile cap geometry is drawn only as context and is not designed.

        **Engineering note**

        This is a preliminary design/check aid. Final design should verify the governing code edition, load combinations,
        reinforcement detailing, local bearing zone effects, shear/friction, seismic provisions, construction joints, crack control,
        and project-specific bridge authority requirements.
        """
    )
