from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from abutment_uls import (  # noqa: E402
    CodeParameters,
    analyze_section,
    combine_bearing_loads,
    default_code_parameters,
    find_reinforcement,
)
from abutment_uls.plots import (  # noqa: E402
    front_view,
    interaction_plot,
    load_vector_plot,
    plan_view,
    reinforcement_plan,
    side_view,
)


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


def default_bearings(count: int, width_x_mm: float, height_z_mm: float) -> pd.DataFrame:
    spacing = width_x_mm / (count + 1)
    rows = []
    for index in range(count):
        x = -width_x_mm / 2.0 + spacing * (index + 1)
        rows.append(
            {
                "name": f"B{index + 1}",
                "x_mm": round(x, 0),
                "y_mm": 0.0,
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
load_cols = st.columns([1, 1, 3])
with load_cols[0]:
    bearing_count = st.number_input("Number of bearings", min_value=1, max_value=40, value=4, step=1)
with load_cols[1]:
    reset_table = st.button("Reset layout", use_container_width=True)

if "bearing_table" not in st.session_state or reset_table:
    st.session_state.bearing_table = default_bearings(int(bearing_count), width_x_mm, height_z_mm)
elif len(st.session_state.bearing_table) != int(bearing_count):
    st.session_state.bearing_table = default_bearings(int(bearing_count), width_x_mm, height_z_mm)

edited = st.data_editor(
    st.session_state.bearing_table,
    num_rows="dynamic",
    use_container_width=True,
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
        st.dataframe(summary, use_container_width=True, hide_index=True)

        rebar_text = (
            f"{check.bar_count} bars DB{check.bar_dia_mm:.0f}: "
            f"{check.bars_x_face} bars on each x-face, "
            f"{check.bars_y_face} bars on each y-face"
        )
        st.info(rebar_text)

    st.plotly_chart(load_vector_plot(resultant.mux_knm, resultant.muy_knm), use_container_width=True)

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
            use_container_width=True,
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
            use_container_width=True,
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
        use_container_width=True,
    )

with tabs[2]:
    if check is None:
        st.info("Section check will appear after a valid reinforcement layout is available.")
    else:
        sec_cols = st.columns(2)
        with sec_cols[0]:
            st.plotly_chart(reinforcement_plan(check), use_container_width=True)
        with sec_cols[1]:
            st.plotly_chart(
                interaction_plot(check, resultant.pu_kn, resultant.design_mux_knm, resultant.design_muy_knm),
                use_container_width=True,
            )
        bar_table = pd.DataFrame(
            [{"bar": idx + 1, "x_mm": bar.x_mm, "y_mm": bar.y_mm, "area_mm2": bar.area_mm2} for idx, bar in enumerate(check.bars)]
        )
        st.dataframe(bar_table, use_container_width=True, hide_index=True)

with tabs[3]:
    st.markdown(
        """
        **Coordinate and sign convention**

        x and y are plan axes. z is vertical upward. Bearing `Pu_z` is entered as positive downward compression.
        `Pu_x`, `Pu_y`, `Mu_x`, and `Mu_y` follow the displayed positive axes and the right-hand rule.

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
