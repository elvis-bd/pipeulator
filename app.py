import streamlit as st
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pipe data
# ---------------------------------------------------------------------------

# Inner diameters in inches, keyed by nominal size string
COPPER_TYPE_L_ID = {
    '1/2"':   0.545,
    '3/4"':   0.785,
    '1"':     1.025,
    '1-1/4"': 1.265,
    '1-1/2"': 1.505,
    '2"':     1.985,
    '2-1/2"': 2.465,
    '3"':     2.945,
    '3-1/2"': 3.425,
    '4"':     3.905,
    '5"':     4.875,
    '6"':     5.845,
}

# PEX SDR-9 per ASTM F876 (same dimensions for PEX-A and PEX-B)
PEX_ID = {
    '1/2"':   0.485,
    '3/4"':   0.681,
    '1"':     0.875,
    '1-1/4"': 1.075,
    '1-1/2"': 1.275,
    '2"':     1.695,
}

# Hazen-Williams C values
HW_C = {
    "Copper (fairly rough)": 130,
    "PEX": 150,
}

# Ordered nominal sizes for display
ALL_SIZES = [
    '1/2"', '3/4"', '1"', '1-1/4"', '1-1/2"', '2"',
    '2-1/2"', '3"', '3-1/2"', '4"', '5"', '6"',
]

# Size to numeric inches for sorting
SIZE_TO_INCHES = {
    '1/2"': 0.5, '3/4"': 0.75, '1"': 1.0, '1-1/4"': 1.25,
    '1-1/2"': 1.5, '2"': 2.0, '2-1/2"': 2.5, '3"': 3.0,
    '3-1/2"': 3.5, '4"': 4.0, '5"': 5.0, '6"': 6.0,
}

# ---------------------------------------------------------------------------
# WSFU ↔ GPM tables per UPC Table A 4.1
# Each entry: (WSFU, flush_tank_gpm, flush_valve_gpm)
# Flush valve column starts at 5 WSFU per the UPC table.
# ---------------------------------------------------------------------------

_WSFU_GPM_TABLE = [
    # (WSFU, flush_tank, flush_valve)
    (1,     1.0,   None),
    (2,     2.0,   None),
    (3,     3.0,   None),
    (4,     4.0,   None),
    (5,     4.5,   22.0),
    (6,     5.0,   23.0),
    (7,     6.0,   24.0),
    (8,     7.0,   25.0),
    (9,     7.5,   26.0),
    (10,    8.0,   27.0),
    (11,    8.5,   28.0),
    (12,    9.0,   29.0),
    (13,    10.0,  29.5),
    (14,    10.5,  30.0),
    (15,    11.0,  31.0),
    (16,    12.0,  32.0),
    (17,    12.5,  33.0),
    (18,    13.0,  33.5),
    (19,    13.5,  34.0),
    (20,    14.0,  35.0),
    (25,    17.0,  38.0),
    (30,    20.0,  41.0),
    (40,    25.0,  47.0),
    (50,    29.0,  51.0),
    (60,    33.0,  55.0),
    (80,    39.0,  62.0),
    (100,   44.0,  68.0),
    (120,   49.0,  74.0),
    (140,   53.0,  78.0),
    (160,   57.0,  83.0),
    (180,   61.0,  87.0),
    (200,   65.0,  91.0),
    (225,   70.0,  95.0),
    (250,   75.0,  100.0),
    (300,   85.0,  110.0),
    (400,   105.0, 125.0),
    (500,   125.0, 140.0),
    (750,   170.0, 175.0),
    (1000,  210.0, 210.0),
    (1250,  240.0, 240.0),
    (1500,  270.0, 270.0),
    (1750,  300.0, 300.0),
    (2000,  325.0, 325.0),
    (2500,  380.0, 380.0),
    (3000,  435.0, 435.0),
    (4000,  525.0, 525.0),
    (5000,  600.0, 600.0),
    (6000,  650.0, 650.0),
    (7000,  700.0, 700.0),
    (8000,  730.0, 730.0),
    (9000,  760.0, 760.0),
    (10000, 790.0, 790.0),
]

# Build separate arrays for flush tank and flush valve
_WSFU_VALS_TANK = np.array([w for w, _, _ in _WSFU_GPM_TABLE], dtype=float)
_GPM_VALS_TANK = np.array([g for _, g, _ in _WSFU_GPM_TABLE], dtype=float)

# Flush valve table starts at 5 WSFU
_FV_ENTRIES = [(w, gv) for w, _, gv in _WSFU_GPM_TABLE if gv is not None]
_WSFU_VALS_VALVE = np.array([w for w, _ in _FV_ENTRIES], dtype=float)
_GPM_VALS_VALVE = np.array([g for _, g in _FV_ENTRIES], dtype=float)


def gpm_to_wsfu(gpm: float, fixture_type: str = "Flush tank") -> float:
    """Convert GPM demand to WSFU using inverse interpolation of UPC table."""
    if fixture_type == "Flush valve":
        wsfu_arr, gpm_arr = _WSFU_VALS_VALVE, _GPM_VALS_VALVE
    else:
        wsfu_arr, gpm_arr = _WSFU_VALS_TANK, _GPM_VALS_TANK

    if gpm <= 0:
        return 0.0
    if gpm <= gpm_arr[0]:
        return wsfu_arr[0] * gpm / gpm_arr[0]
    if gpm >= gpm_arr[-1]:
        return float(wsfu_arr[-1])
    return float(np.interp(gpm, gpm_arr, wsfu_arr))


# ---------------------------------------------------------------------------
# Hydraulic calculations (Hazen-Williams per UPC Appendix A)
# ---------------------------------------------------------------------------

def max_gpm_by_pressure(dp_per_100ft: float, C: float, d_in: float) -> float:
    """
    Max GPM for a given uniform pressure-drop limit.

    Hazen-Williams:  ΔP (psi/ft) = 4.52 × Q^1.852 / (C^1.852 × d^4.8704)

    Solved for Q:
        Q = ((ΔP × C^1.852 × d^4.8704) / 4.52) ^ (1/1.852)

    dp_per_100ft : allowable pressure drop in psi per 100 ft
    """
    dp_per_ft = dp_per_100ft / 100.0
    numerator = dp_per_ft * (C ** 1.852) * (d_in ** 4.8704)
    return (numerator / 4.52) ** (1.0 / 1.852)


def max_gpm_by_velocity(v_fps: float, d_in: float) -> float:
    """Max GPM limited by velocity.  v = 0.4085 × Q / d² ⟹ Q = v × d² / 0.4085"""
    return v_fps * d_in ** 2 / 0.4085


def velocity_at_gpm(gpm: float, d_in: float) -> float:
    """Return velocity in ft/s for a given flow and inside diameter."""
    return 0.4085 * gpm / d_in ** 2


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Pipeulator", layout="wide")
st.title("Pipeulator — UPC Appendix A Pipe Sizing")
st.caption("Calculates allowable flow per pipe size using UPC uniform pressure-drop methodology (Hazen-Williams)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Inputs")

    material = st.selectbox("Pipe material", list(HW_C.keys()))

    dp_limit = st.number_input(
        "Allowable pressure drop (psi per 100 ft)",
        min_value=0.1, max_value=50.0, value=4.0, step=0.5,
    )

    fixture_type = st.radio("Fixture type", ["Flush tank", "Flush valve"], horizontal=True)

    water_type = st.radio("Water type", ["Cold", "Hot"], horizontal=True)

    hot_temp = None
    if water_type == "Hot":
        hot_temp = st.number_input(
            "Hot water temperature (°F)",
            min_value=100, max_value=210, value=120, step=5,
        )

    # Velocity limits per UPC
    if water_type == "Cold":
        v_limit = 8.0
    elif hot_temp is not None and hot_temp >= 140:
        v_limit = 3.0
    else:
        v_limit = 5.0

    st.info(f"Velocity limit applied: **{v_limit} ft/s** "
            f"({'cold water' if water_type == 'Cold' else f'hot water @ {hot_temp}°F'})")

    max_size = st.selectbox(
        "Maximum pipe size to display",
        ALL_SIZES,
        index=ALL_SIZES.index('6"') if material == "Copper (fairly rough)" else ALL_SIZES.index('2"'),
    )

    if material == "PEX":
        st.caption("PEX tubing is commonly available up to 2\". Sizes above 2\" are not shown for PEX.")

# Determine ID lookup and available sizes
if material == "Copper (fairly rough)":
    id_lookup = COPPER_TYPE_L_ID
else:
    id_lookup = PEX_ID

C = HW_C[material]
max_size_in = SIZE_TO_INCHES[max_size]

# Build results
rows = []
for size in ALL_SIZES:
    if SIZE_TO_INCHES[size] > max_size_in:
        break
    if size not in id_lookup:
        continue
    d = id_lookup[size]

    q_pressure = max_gpm_by_pressure(dp_limit, C, d)
    q_velocity = max_gpm_by_velocity(v_limit, d)
    q_allowed = min(q_pressure, q_velocity)
    limiting = "Velocity" if q_velocity < q_pressure else "Pressure drop"
    v_actual = velocity_at_gpm(q_allowed, d)
    wsfu_type = "Flush tank" if water_type == "Hot" else fixture_type
    wsfu = gpm_to_wsfu(q_allowed, wsfu_type)

    rows.append({
        "Nominal Size": size,
        "ID (in)": f"{d:.3f}",
        "Max GPM (pressure)": f"{q_pressure:.1f}",
        "Max GPM (velocity)": f"{q_velocity:.1f}",
        "Allowed GPM": f"{q_allowed:.1f}",
        "Velocity (ft/s)": f"{v_actual:.1f}",
        "Fixture Units (WSFU)": f"{wsfu:.0f}",
        "Limiting Factor": limiting,
    })

with col2:
    st.subheader("Results")
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.warning("No pipe sizes available for the selected material and range.")

# ---------------------------------------------------------------------------
# Quick single-pipe lookup
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Quick Lookup — Check a specific flow")

qcol1, qcol2, qcol3 = st.columns(3)
with qcol1:
    lookup_gpm = st.number_input("Flow rate (GPM)", min_value=0.1, max_value=1000.0, value=10.0, step=1.0)
with qcol2:
    wsfu_lookup_type = "Flush tank" if water_type == "Hot" else fixture_type
    st.metric("Equivalent WSFU", f"{gpm_to_wsfu(lookup_gpm, wsfu_lookup_type):.0f}")
with qcol3:
    if rows:
        # Find smallest adequate pipe
        best = None
        for r in rows:
            if float(r["Allowed GPM"]) >= lookup_gpm:
                best = r["Nominal Size"]
                break
        if best:
            st.metric("Minimum pipe size", best)
        else:
            st.metric("Minimum pipe size", "Exceeds range")

st.divider()
st.caption(
    "Based on UPC Appendix A sizing methodology. "
    "Hazen-Williams C values: Copper (fairly rough) = 130, PEX = 150. "
    "Pipe IDs: Copper Type L per ASTM B88, PEX per ASTM F876 SDR-9. "
    "WSFU-to-GPM conversion per UPC Table A 4.1 (flush-tank and flush-valve systems). "
    "Velocity limits per UPC: 8 fps cold, 5 fps hot (<140°F), 3 fps hot (≥140°F)."
)
