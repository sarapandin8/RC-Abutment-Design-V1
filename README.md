# RC Bridge Abutment ULS Designer

Streamlit web app for preliminary ULS design/check of a rectangular reinforced concrete bridge abutment section.

The app lets the user define bearing count, bearing locations, and factored actions at each bearing:

- `Pu_x`, `Pu_y`, `Pu_z`
- `Mu_x`, `Mu_y`

It combines those actions to fixed-base resultants, shows plan/front/side views with x-y-z axes, draws the display-only pile cap, and checks the base section for axial load plus biaxial bending.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy To Streamlit Community Cloud

Use this repository with:

- Main file path: `app.py`
- Python dependencies: `requirements.txt`

## Engineering Scope

Coordinate convention:

- x and y are plan axes.
- z is vertical upward.
- `Pu_z` is entered as positive downward compression.
- `Pu_x`, `Pu_y`, `Mu_x`, and `Mu_y` follow the displayed axes/right-hand rule.

Fixed-base resultants:

```text
Pu  = sum(Pu_z)
Vx  = sum(Pu_x)
Vy  = sum(Pu_y)
Mux = sum(Mu_x + (-y Pu_z - z Pu_y) / 1000)
Muy = sum(Mu_y + ( z Pu_x + x Pu_z) / 1000)
Tz  = sum((x Pu_y - y Pu_x) / 1000)
```

Units are kN, kN-m, MPa, and mm.

The section check uses:

- Whitney rectangular stress block
- Strain compatibility for uniaxial `P-Mx` and `P-My` curves
- Editable resistance-factor defaults for ACI-style and AASHTO-LRFD-style checks
- Conservative linear biaxial interaction:

```text
|Mux| / phi Mnx(Pu) + |Muy| / phi Mny(Pu) <= 1.0
```

## Important Limitations

This app is a preliminary design/check aid, not a sealed design calculation. Final engineering should verify the governing code edition and project requirements, including:

- load combinations and load factors
- local bearing zone design
- shear, friction, torsion, and construction joint transfer
- seismic provisions
- reinforcement detailing and anchorage
- crack control and service checks
- pile cap and pile design

Public reference starting points:

- FHWA LRFD bridge design examples: <https://www.fhwa.dot.gov/bridge/lrfd/>
- ACI 318 code information: <https://www.concrete.org/store/productdetail.aspx?ItemID=318U19>
