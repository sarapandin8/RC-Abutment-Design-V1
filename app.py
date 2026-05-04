from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


Axis = Literal["x", "y"]


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
    bars_x_face: int
    bars_y_face: int
    phi_pmax_kn: float
    phi_mnx_at_pu_knm: float
    phi_mny_at_pu_knm: float
    axial_ratio: float
    mux_ratio: float
    muy_ratio: float
    biaxial_ratio: float
    governing_ratio: float
    status: str
    curve_x: list[dict[str, float]]
    curve_y: list[dict[str, float]]
    bars: list[Bar]


def default_code_parameters(code_name: str) -> CodeParameters:
    if "AASHTO" in code_name.upper():
        return CodeParameters(
            name="AASHTO LRFD style",
            phi_compression=0.75,
            phi_flexure=0.90,
            phi_method="aashto_axial",
            axial_cap_factor=1.00,
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
        force_n = stress * bar.area_mm2
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
    biaxial_ratio = mux_ratio + muy_ratio
    governing_ratio = max(axial_ratio, biaxial_ratio)
    status = "OK" if governing_ratio <= 1.0 else "NG"

    return SectionCheck(
        width_x_mm=width_x_mm,
        depth_y_mm=depth_y_mm,
        as_total_mm2=as_total,
        rho_percent=rho,
        bar_count=len(bars),
        bar_dia_mm=bar_dia_mm,
        bars_x_face=bars_x_face,
        bars_y_face=bars_y_face,
        phi_pmax_kn=pmax,
        phi_mnx_at_pu_knm=cap_x,
        phi_mny_at_pu_knm=cap_y,
        axial_ratio=axial_ratio,
        mux_ratio=mux_ratio,
        muy_ratio=muy_ratio,
        biaxial_ratio=biaxial_ratio,
        governing_ratio=governing_ratio,
        status=status,
        curve_x=curve_x,
        curve_y=curve_y,
        bars=bars,
    )


def find_reinforcement(
    *,
    width_x_mm: float,
    depth_y_mm: float,
    cover_mm: float,
    bar_dia_options_mm: Iterable[float],
    fc_mpa: float,
    fy_mpa: float,
    es_mpa: float,
    pu_kn: float,
    mux_knm: float,
    muy_knm: float,
    params: CodeParameters,
    rho_min_percent: float,
    rho_max_percent: float,
    max_bars_x_face: int = 28,
    max_bars_y_face: int = 18,
    sample_count: int = 120,
) -> SectionCheck | None:
    candidates: list[tuple[float, float, int, int]] = []
    ag = width_x_mm * depth_y_mm
    for dia in bar_dia_options_mm:
        area = bar_area_mm2(dia)
        for nx in range(2, max_bars_x_face + 1):
            for ny in range(2, max_bars_y_face + 1):
                bar_count = 2 * nx + 2 * max(0, ny - 2)
                as_total = bar_count * area
                rho = as_total / ag * 100.0
                if rho_min_percent <= rho <= rho_max_percent:
                    candidates.append((as_total, dia, nx, ny))

    for _, dia, nx, ny in sorted(candidates, key=lambda item: item[0]):
        try:
            check = analyze_section(
                width_x_mm=width_x_mm,
                depth_y_mm=depth_y_mm,
                cover_mm=cover_mm,
                bar_dia_mm=dia,
                bars_x_face=nx,
                bars_y_face=ny,
                fc_mpa=fc_mpa,
                fy_mpa=fy_mpa,
                es_mpa=es_mpa,
                pu_kn=pu_kn,
                mux_knm=mux_knm,
                muy_knm=muy_knm,
                params=params,
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
) -> None:
    fig.add_shape(
        type="rect",
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
        text=label,
        showarrow=True,
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor=color,
        font={"color": color, "size": 13},
        bgcolor="rgba(255,255,255,0.75)",
        borderpad=2,
    )


def _finish_view(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
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
        zeroline=True,
        zerolinecolor="#111827",
        zerolinewidth=1,
    )
    fig.update_yaxes(
        title=y_title,
        showgrid=True,
        gridcolor=COLORS["grid"],
        zeroline=True,
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
    bearing_size_mm: float,
) -> go.Figure:
    fig = go.Figure()
    pile_x = width_x_mm / 2.0 + pilecap_overhang_mm
    pile_y = depth_y_mm / 2.0 + pilecap_overhang_mm
    abut_x = width_x_mm / 2.0
    abut_y = depth_y_mm / 2.0

    _add_rect(fig, x0=-pile_x, x1=pile_x, y0=-pile_y, y1=pile_y, fillcolor=COLORS["pilecap"], linecolor=COLORS["pilecap_line"], dash="dash", opacity=0.85)
    _add_rect(fig, x0=-abut_x, x1=abut_x, y0=-abut_y, y1=abut_y, fillcolor=COLORS["concrete"], linecolor=COLORS["concrete_line"])

    half = bearing_size_mm / 2.0
    for row in bearings:
        x = float(row.get("x_mm", 0.0))
        y = float(row.get("y_mm", 0.0))
        name = str(row.get("name", "B"))
        _add_rect(fig, x0=x - half, x1=x + half, y0=y - half, y1=y + half, fillcolor=COLORS["bearing"], linecolor="#065f5b", opacity=0.95)
        fig.add_annotation(x=x, y=y, text=name, showarrow=False, font={"color": "white", "size": 11})

    axis_origin_x = -pile_x * 0.86
    axis_origin_y = -pile_y * 0.86
    arrow = max(width_x_mm, depth_y_mm) * 0.22
    _add_axis_arrow(fig, x=axis_origin_x, y=axis_origin_y, dx=arrow, dy=0, label="+x", color=COLORS["axis_x"])
    _add_axis_arrow(fig, x=axis_origin_x, y=axis_origin_y, dx=0, dy=arrow, label="+y", color=COLORS["axis_y"])

    pad = max(width_x_mm, depth_y_mm) * 0.12
    fig.update_xaxes(range=[-pile_x - pad, pile_x + pad])
    fig.update_yaxes(range=[-pile_y - pad, pile_y + pad])
    return _finish_view(fig, "Section plan at bearing level", "x (mm)", "y (mm)")


def front_view(
    bearings: Iterable[dict],
    *,
    width_x_mm: float,
    height_z_mm: float,
    pilecap_overhang_mm: float,
    pilecap_thickness_mm: float,
    bearing_size_mm: float,
) -> go.Figure:
    fig = go.Figure()
    pile_x = width_x_mm / 2.0 + pilecap_overhang_mm
    abut_x = width_x_mm / 2.0
    half = bearing_size_mm / 2.0
    bearing_h = max(80.0, bearing_size_mm * 0.28)

    _add_rect(fig, x0=-pile_x, x1=pile_x, y0=-pilecap_thickness_mm, y1=0, fillcolor=COLORS["pilecap"], linecolor=COLORS["pilecap_line"], dash="dash", opacity=0.85)
    _add_rect(fig, x0=-abut_x, x1=abut_x, y0=0, y1=height_z_mm, fillcolor=COLORS["concrete"], linecolor=COLORS["concrete_line"])

    for row in bearings:
        x = float(row.get("x_mm", 0.0))
        z = float(row.get("z_mm", height_z_mm))
        name = str(row.get("name", "B"))
        _add_rect(fig, x0=x - half, x1=x + half, y0=z, y1=z + bearing_h, fillcolor=COLORS["bearing"], linecolor="#065f5b")
        fig.add_annotation(x=x, y=z + bearing_h / 2.0, text=name, showarrow=False, font={"color": "white", "size": 11})

    axis_origin_x = -pile_x * 0.86
    axis_origin_z = -pilecap_thickness_mm * 0.72
    arrow = max(width_x_mm, height_z_mm) * 0.16
    _add_axis_arrow(fig, x=axis_origin_x, y=axis_origin_z, dx=arrow, dy=0, label="+x", color=COLORS["axis_x"])
    _add_axis_arrow(fig, x=axis_origin_x, y=axis_origin_z, dx=0, dy=arrow, label="+z", color=COLORS["axis_z"])

    pad = max(width_x_mm, height_z_mm) * 0.10
    fig.update_xaxes(range=[-pile_x - pad, pile_x + pad])
    fig.update_yaxes(range=[-pilecap_thickness_mm - pad * 0.35, height_z_mm + bearing_h + pad * 0.35])
    return _finish_view(fig, "Front view", "x (mm)", "z (mm)")


def side_view(
    bearings: Iterable[dict],
    *,
    depth_y_mm: float,
    height_z_mm: float,
    pilecap_overhang_mm: float,
    pilecap_thickness_mm: float,
    bearing_size_mm: float,
) -> go.Figure:
    fig = go.Figure()
    pile_y = depth_y_mm / 2.0 + pilecap_overhang_mm
    abut_y = depth_y_mm / 2.0
    half = bearing_size_mm / 2.0
    bearing_h = max(80.0, bearing_size_mm * 0.28)

    _add_rect(fig, x0=-pile_y, x1=pile_y, y0=-pilecap_thickness_mm, y1=0, fillcolor=COLORS["pilecap"], linecolor=COLORS["pilecap_line"], dash="dash", opacity=0.85)
    _add_rect(fig, x0=-abut_y, x1=abut_y, y0=0, y1=height_z_mm, fillcolor=COLORS["concrete"], linecolor=COLORS["concrete_line"])

    for row in bearings:
        y = float(row.get("y_mm", 0.0))
        z = float(row.get("z_mm", height_z_mm))
        name = str(row.get("name", "B"))
        _add_rect(fig, x0=y - half, x1=y + half, y0=z, y1=z + bearing_h, fillcolor=COLORS["bearing"], linecolor="#065f5b")
        fig.add_annotation(x=y, y=z + bearing_h / 2.0, text=name, showarrow=False, font={"color": "white", "size": 11})

    axis_origin_y = -pile_y * 0.86
    axis_origin_z = -pilecap_thickness_mm * 0.72
    arrow = max(depth_y_mm, height_z_mm) * 0.16
    _add_axis_arrow(fig, x=axis_origin_y, y=axis_origin_z, dx=arrow, dy=0, label="+y", color=COLORS["axis_y"])
    _add_axis_arrow(fig, x=axis_origin_y, y=axis_origin_z, dx=0, dy=arrow, label="+z", color=COLORS["axis_z"])

    pad = max(depth_y_mm, height_z_mm) * 0.10
    fig.update_xaxes(range=[-pile_y - pad, pile_y + pad])
    fig.update_yaxes(range=[-pilecap_thickness_mm - pad * 0.35, height_z_mm + bearing_h + pad * 0.35])
    return _finish_view(fig, "Side view", "y (mm)", "z (mm)")


def reinforcement_plan(check: SectionCheck) -> go.Figure:
    fig = go.Figure()
    x = check.width_x_mm / 2.0
    y = check.depth_y_mm / 2.0
    _add_rect(fig, x0=-x, x1=x, y0=-y, y1=y, fillcolor="#f8fafc", linecolor=COLORS["concrete_line"])
    marker_size = max(7, min(16, check.bar_dia_mm * 0.45))
    fig.add_trace(
        go.Scatter(
            x=[bar.x_mm for bar in check.bars],
            y=[bar.y_mm for bar in check.bars],
            mode="markers",
            marker={"size": marker_size, "color": COLORS["steel"], "line": {"color": "#7c2d12", "width": 1}},
            hovertemplate="x=%{x:.0f} mm<br>y=%{y:.0f} mm<extra></extra>",
        )
    )
    arrow = max(check.width_x_mm, check.depth_y_mm) * 0.18
    _add_axis_arrow(fig, x=-x * 0.78, y=-y * 0.78, dx=arrow, dy=0, label="+x", color=COLORS["axis_x"])
    _add_axis_arrow(fig, x=-x * 0.78, y=-y * 0.78, dx=0, dy=arrow, label="+y", color=COLORS["axis_y"])
    pad = max(check.width_x_mm, check.depth_y_mm) * 0.08
    fig.update_xaxes(range=[-x - pad, x + pad])
    fig.update_yaxes(range=[-y - pad, y + pad])
    return _finish_view(fig, "Base section reinforcement", "x (mm)", "y (mm)")


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


def load_vector_plot(resultant_mx: float, resultant_my: float) -> go.Figure:
    magnitude = math.hypot(resultant_mx, resultant_my)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, resultant_my],
            y=[0, resultant_mx],
            mode="lines+markers",
            line={"color": "#334155", "width": 4},
            marker={"size": [8, 12], "color": ["#334155", "#dc2626"]},
            hovertemplate="My=%{x:.1f} kN-m<br>Mx=%{y:.1f} kN-m<extra></extra>",
        )
    )
    fig.add_annotation(x=resultant_my, y=resultant_mx, text=f"|M| = {magnitude:,.0f} kN-m", showarrow=True, arrowhead=2, ax=-25, ay=-25)
    fig.update_layout(
        title={"text": "Signed base moment vector", "x": 0.02, "xanchor": "left"},
        height=340,
        margin={"l": 24, "r": 24, "t": 54, "b": 24},
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        font={"family": "Arial, sans-serif", "size": 13, "color": "#172033"},
    )
    fig.update_xaxes(title="My (kN-m)", gridcolor=COLORS["grid"], zeroline=True, zerolinecolor="#111827")
    fig.update_yaxes(title="Mx (kN-m)", gridcolor=COLORS["grid"], zeroline=True, zerolinecolor="#111827")
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


def status_html(status: str, ratio: float) -> str:
    klass = "status-ok" if status == "OK" else "status-ng"
    label = "PASS" if status == "OK" else "FAIL"
    return f'<span class="status-pill {klass}">{label} &nbsp; U = {ratio:.3f}</span>'


def metric_row(resultant, check):
    cols = st.columns(6)
    cols[0].metric("Pu compression", f"{resultant.pu_kn:,.0f} kN")
    cols[1].metric("Vx resultant", f"{resultant.vx_kn:,.0f} kN")
    cols[2].metric("Vy resultant", f"{resultant.vy_kn:,.0f} kN")
    cols[3].metric("Mux design", f"{resultant.design_mux_knm:,.0f} kN-m")
    cols[4].metric("Muy design", f"{resultant.design_muy_knm:,.0f} kN-m")
    cols[5].metric("Tz resultant", f"{resultant.torsion_z_knm:,.0f} kN-m")

    if check is not None:
        cols = st.columns(5)
        cols[0].metric("phi Pmax", f"{check.phi_pmax_kn:,.0f} kN")
        cols[1].metric("phi Mnx at Pu", f"{check.phi_mnx_at_pu_knm:,.0f} kN-m")
        cols[2].metric("phi Mny at Pu", f"{check.phi_mny_at_pu_knm:,.0f} kN-m")
        cols[3].metric("As provided", f"{check.as_total_mm2:,.0f} mm2")
        cols[4].metric("rho", f"{check.rho_percent:.3f} %")


st.title("RC Bridge Abutment ULS Designer")
st.caption("Bearing load resultants, fixed-base axial + biaxial bending check, and rectangular section visuals.")

with st.sidebar:
    st.header("Design Basis")
    code_choice = st.selectbox(
        "Code assumptions",
        ["ACI 318 style", "AASHTO LRFD style"],
        help="Resistance-factor defaults are editable below. Always verify against the governing project code edition.",
    )
    base_params = default_code_parameters(code_choice)

    fc_mpa = st.number_input("f'c (MPa)", min_value=15.0, max_value=100.0, value=35.0, step=1.0)
    fy_mpa = st.number_input("fy (MPa)", min_value=240.0, max_value=700.0, value=420.0, step=10.0)
    es_mpa = st.number_input("Es (MPa)", min_value=180000.0, max_value=220000.0, value=200000.0, step=5000.0)

    with st.expander("Resistance factor settings", expanded=False):
        phi_compression = st.number_input(
            "phi compression",
            min_value=0.40,
            max_value=0.95,
            value=float(base_params.phi_compression),
            step=0.01,
        )
        phi_flexure = st.number_input(
            "phi flexure / tension",
            min_value=0.40,
            max_value=0.95,
            value=float(base_params.phi_flexure),
            step=0.01,
        )
        axial_cap_factor = st.number_input(
            "axial cap factor",
            min_value=0.50,
            max_value=1.00,
            value=float(base_params.axial_cap_factor),
            step=0.01,
        )
        eps_cu = st.number_input("concrete ultimate strain", min_value=0.0020, max_value=0.0040, value=0.0030, step=0.0001, format="%.4f")

    params = replace(
        base_params,
        phi_compression=phi_compression,
        phi_flexure=phi_flexure,
        axial_cap_factor=axial_cap_factor,
        eps_cu=eps_cu,
    )

    st.header("Geometry")
    width_x_mm = st.number_input("Abutment width along x (mm)", min_value=800.0, value=9000.0, step=100.0)
    depth_y_mm = st.number_input("Abutment thickness along y (mm)", min_value=300.0, value=1200.0, step=50.0)
    height_z_mm = st.number_input("Bearing level height z (mm)", min_value=500.0, value=4500.0, step=100.0)
    bearing_size_mm = st.number_input("Bearing plan size (mm)", min_value=100.0, value=250.0, step=25.0)
    pilecap_overhang_mm = st.number_input("Pile cap overhang each side (mm)", min_value=0.0, value=500.0, step=50.0)
    pilecap_thickness_mm = st.number_input("Pile cap display thickness (mm)", min_value=300.0, value=1500.0, step=100.0)

    st.header("Reinforcement")
    cover_mm = st.number_input("Clear cover to tie / outer bar (mm)", min_value=25.0, value=75.0, step=5.0)
    mode = st.radio("Reinforcement mode", ["Auto design", "Manual check"], horizontal=True)
    rho_min_percent = st.number_input("minimum rho for auto (%)", min_value=0.0, max_value=5.0, value=0.25, step=0.05)
    rho_max_percent = st.number_input("maximum rho for auto (%)", min_value=0.1, max_value=10.0, value=4.00, step=0.10)

    if mode == "Manual check":
        bar_dia_mm = st.selectbox("bar diameter (mm)", [16.0, 20.0, 25.0, 28.0, 32.0, 36.0], index=2)
        bars_x_face = st.number_input("bars on each x-face", min_value=2, max_value=40, value=12, step=1)
        bars_y_face = st.number_input("bars on each y-face", min_value=2, max_value=24, value=3, step=1)
    else:
        dia_options = st.multiselect(
            "auto bar diameters (mm)",
            [16.0, 20.0, 25.0, 28.0, 32.0, 36.0],
            default=[20.0, 25.0, 28.0, 32.0],
        )


st.subheader("Bearing Loads")
load_cols = st.columns([1, 1, 1, 1])
with load_cols[0]:
    bearing_rows = st.radio("Bearing rows", [1, 2], horizontal=True)
with load_cols[1]:
    bearings_per_row = st.number_input("Bearings per row", min_value=1, max_value=20, value=4, step=1)
with load_cols[2]:
    row_spacing_y_mm = st.number_input(
        "Row spacing y (mm)",
        min_value=0.0,
        max_value=float(depth_y_mm),
        value=min(600.0, float(depth_y_mm)),
        step=50.0,
        disabled=int(bearing_rows) == 1,
    )
with load_cols[3]:
    reset_table = st.button("Reset layout", width="stretch")

expected_bearing_count = int(bearing_rows) * int(bearings_per_row)

if "bearing_table" not in st.session_state or reset_table:
    st.session_state.bearing_table = default_bearings(
        int(bearings_per_row),
        int(bearing_rows),
        width_x_mm,
        height_z_mm,
        row_spacing_y_mm,
    )
elif len(st.session_state.bearing_table) != expected_bearing_count:
    st.session_state.bearing_table = default_bearings(
        int(bearings_per_row),
        int(bearing_rows),
        width_x_mm,
        height_z_mm,
        row_spacing_y_mm,
    )

edited = st.data_editor(
    st.session_state.bearing_table,
    num_rows="dynamic",
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
bearings_df = clean_bearings(edited)
st.session_state.bearing_table = bearings_df
records = bearings_df.to_dict("records")
resultant = combine_bearing_loads(records)

if resultant.pu_kn < 0:
    st.warning("Pu_z resultant is net tension. The app can show resultants, but reinforcement design assumptions should be checked carefully.")

check = None
design_error = None
try:
    if mode == "Auto design":
        if not dia_options:
            design_error = "Select at least one bar diameter for auto design."
        else:
            with st.spinner("Searching reinforcement layout..."):
                check = find_reinforcement(
                    width_x_mm=width_x_mm,
                    depth_y_mm=depth_y_mm,
                    cover_mm=cover_mm,
                    bar_dia_options_mm=tuple(dia_options),
                    fc_mpa=fc_mpa,
                    fy_mpa=fy_mpa,
                    es_mpa=es_mpa,
                    pu_kn=resultant.pu_kn,
                    mux_knm=resultant.design_mux_knm,
                    muy_knm=resultant.design_muy_knm,
                    params=params,
                    rho_min_percent=rho_min_percent,
                    rho_max_percent=rho_max_percent,
                )
            if check is None:
                design_error = "No reinforcement layout passed within the selected auto limits."
    else:
        check = analyze_section(
            width_x_mm=width_x_mm,
            depth_y_mm=depth_y_mm,
            cover_mm=cover_mm,
            bar_dia_mm=float(bar_dia_mm),
            bars_x_face=int(bars_x_face),
            bars_y_face=int(bars_y_face),
            fc_mpa=fc_mpa,
            fy_mpa=fy_mpa,
            es_mpa=es_mpa,
            pu_kn=resultant.pu_kn,
            mux_knm=resultant.design_mux_knm,
            muy_knm=resultant.design_muy_knm,
            params=params,
        )
except Exception as exc:  # noqa: BLE001
    design_error = str(exc)


tabs = st.tabs(["Results", "Views", "Section", "Method"])

with tabs[0]:
    metric_row(resultant, check)
    st.markdown(
        '<p class="small-note">Vx and Vy are reported as fixed-base force resultants only. Shear design is intentionally outside this scope.</p>',
        unsafe_allow_html=True,
    )
    if design_error:
        st.error(design_error)
    elif check is not None:
        st.markdown(status_html(check.status, check.governing_ratio), unsafe_allow_html=True)
        st.progress(min(1.0, max(0.0, check.governing_ratio)))
        summary = pd.DataFrame(
            [
                ["Axial Pu / phi Pmax", check.axial_ratio],
                ["Mux / phi Mnx(Pu)", check.mux_ratio],
                ["Muy / phi Mny(Pu)", check.muy_ratio],
                ["Linear biaxial interaction", check.biaxial_ratio],
                ["Governing utilization", check.governing_ratio],
            ],
            columns=["Check", "Ratio"],
        )
        st.dataframe(summary, width="stretch", hide_index=True)

        rebar_text = (
            f"{check.bar_count} bars DB{check.bar_dia_mm:.0f}: "
            f"{check.bars_x_face} bars on each x-face, "
            f"{check.bars_y_face} bars on each y-face"
        )
        st.info(rebar_text)

    st.plotly_chart(load_vector_plot(resultant.mux_knm, resultant.muy_knm), width="stretch")

with tabs[1]:
    view_cols = st.columns(2)
    with view_cols[0]:
        st.plotly_chart(
            plan_view(
                records,
                width_x_mm=width_x_mm,
                depth_y_mm=depth_y_mm,
                pilecap_overhang_mm=pilecap_overhang_mm,
                bearing_size_mm=bearing_size_mm,
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
                bearing_size_mm=bearing_size_mm,
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
            bearing_size_mm=bearing_size_mm,
        ),
        width="stretch",
    )

with tabs[2]:
    if check is None:
        st.info("Section check will appear after a valid reinforcement layout is available.")
    else:
        sec_cols = st.columns(2)
        with sec_cols[0]:
            st.plotly_chart(reinforcement_plan(check), width="stretch")
        with sec_cols[1]:
            st.plotly_chart(
                interaction_plot(check, resultant.pu_kn, resultant.design_mux_knm, resultant.design_muy_knm),
                width="stretch",
            )
        bar_table = pd.DataFrame(
            [{"bar": idx + 1, "x_mm": bar.x_mm, "y_mm": bar.y_mm, "area_mm2": bar.area_mm2} for idx, bar in enumerate(check.bars)]
        )
        st.dataframe(bar_table, width="stretch", hide_index=True)

with tabs[3]:
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

        **RC section check**

        The app uses Whitney stress block strain compatibility for uniaxial `P-Mx` and `P-My` curves.
        Biaxial bending is checked with a conservative linear interaction:

        `|Mux| / phi Mnx(Pu) + |Muy| / phi Mny(Pu) <= 1.0`

        ACI style uses strain-based phi interpolation. AASHTO LRFD style uses editable defaults with axial-to-flexural phi interpolation.
        Pile cap geometry is drawn only as context and is not designed.

        **Engineering note**

        This is a preliminary design/check aid. Final design should verify the governing code edition, load combinations,
        reinforcement detailing, local bearing zone effects, shear/friction, seismic provisions, construction joints, crack control,
        and project-specific bridge authority requirements.
        """
    )
