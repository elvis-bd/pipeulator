import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Pipeulator", layout="wide")

# ---------------------------------------------------------------------------
# Pipe data
# ---------------------------------------------------------------------------

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

PEX_ID = {
    '1/2"':   0.485,
    '3/4"':   0.681,
    '1"':     0.875,
    '1-1/4"': 1.075,
    '1-1/2"': 1.275,
    '2"':     1.695,
}

HW_C = {
    "Copper (fairly rough)": 130,
    "PEX": 150,
}

ALL_SIZES = [
    '1/2"', '3/4"', '1"', '1-1/4"', '1-1/2"', '2"',
    '2-1/2"', '3"', '3-1/2"', '4"', '5"', '6"',
]

SIZE_TO_INCHES = {
    '1/2"': 0.5, '3/4"': 0.75, '1"': 1.0, '1-1/4"': 1.25,
    '1-1/2"': 1.5, '2"': 2.0, '2-1/2"': 2.5, '3"': 3.0,
    '3-1/2"': 3.5, '4"': 4.0, '5"': 5.0, '6"': 6.0,
}

# ---------------------------------------------------------------------------
# Fixture unit table per UPC Table 6-4
# Each entry: (name, cold_wsfu, hot_wsfu, total_wsfu)
# ---------------------------------------------------------------------------

FIXTURE_TABLE = [
    ("Bathtub / Tub-shower combo",  1.5, 1.5, 4.0),
    ("Shower",                      1.5, 1.5, 4.0),
    ("Lavatory",                    0.75, 0.75, 2.0),
    ("Kitchen sink",                1.5, 1.5, 4.0),
    ("Dishwasher (residential)",    0.0, 1.4, 1.4),
    ("Clothes washer",              1.5, 1.5, 4.0),
    ("Laundry sink",                1.5, 1.5, 4.0),
    ("Bar sink",                    1.0, 1.0, 2.0),
    ("Water closet (flush tank)",   2.5, 0.0, 2.5),
    ("Water closet (flush valve)",  10.0, 0.0, 10.0),
    ("Urinal (flush tank)",         3.0, 0.0, 3.0),
    ("Urinal (flush valve)",        5.0, 0.0, 5.0),
    ("Hose bibb",                   5.0, 0.0, 5.0),
]

# ---------------------------------------------------------------------------
# WSFU <-> GPM tables per UPC Table A 4.1
# ---------------------------------------------------------------------------

_WSFU_GPM_TABLE = [
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

_WSFU_VALS_TANK = np.array([w for w, _, _ in _WSFU_GPM_TABLE], dtype=float)
_GPM_VALS_TANK = np.array([g for _, g, _ in _WSFU_GPM_TABLE], dtype=float)

_FV_ENTRIES = [(w, gv) for w, _, gv in _WSFU_GPM_TABLE if gv is not None]
_WSFU_VALS_VALVE = np.array([w for w, _ in _FV_ENTRIES], dtype=float)
_GPM_VALS_VALVE = np.array([g for _, g in _FV_ENTRIES], dtype=float)


def gpm_to_wsfu(gpm: float, fixture_type: str = "Flush tank") -> float:
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


def wsfu_to_gpm(wsfu: float, fixture_type: str = "Flush tank") -> float:
    if fixture_type == "Flush valve":
        wsfu_arr, gpm_arr = _WSFU_VALS_VALVE, _GPM_VALS_VALVE
    else:
        wsfu_arr, gpm_arr = _WSFU_VALS_TANK, _GPM_VALS_TANK
    if wsfu <= 0:
        return 0.0
    if wsfu <= wsfu_arr[0]:
        return gpm_arr[0] * wsfu / wsfu_arr[0]
    if wsfu >= wsfu_arr[-1]:
        return float(gpm_arr[-1])
    return float(np.interp(wsfu, wsfu_arr, gpm_arr))


# ---------------------------------------------------------------------------
# Hydraulic calculations
# ---------------------------------------------------------------------------

def max_gpm_by_pressure(dp_per_100ft: float, C: float, d_in: float) -> float:
    dp_per_ft = dp_per_100ft / 100.0
    numerator = dp_per_ft * (C ** 1.852) * (d_in ** 4.8704)
    return (numerator / 4.52) ** (1.0 / 1.852)


def max_gpm_by_velocity(v_fps: float, d_in: float) -> float:
    return v_fps * d_in ** 2 / 0.4085


def velocity_at_gpm(gpm: float, d_in: float) -> float:
    return 0.4085 * gpm / d_in ** 2


# ---------------------------------------------------------------------------
# Page: System Design
# ---------------------------------------------------------------------------

def page_system_design():
    st.title("System Design — Pressure Drop Calculator")
    st.caption("Calculate uniform pressure drop and fixture unit demand per UPC Appendix A")

    col_fix, col_pressure = st.columns([1, 1])

    # --- Fixture counting ---
    with col_fix:
        st.subheader("Fixture Count")
        st.caption("Enter quantity of each fixture type. WSFU per UPC Table 6-4.")

        total_cold = 0.0
        total_hot = 0.0
        total_combined = 0.0

        fixture_rows = []
        for name, cold_per, hot_per, total_per in FIXTURE_TABLE:
            qty = st.number_input(
                name, min_value=0, max_value=500, value=0, step=1,
                key=f"fix_{name}",
            )
            if qty > 0:
                f_cold = cold_per * qty
                f_hot = hot_per * qty
                f_total = total_per * qty
                total_cold += f_cold
                total_hot += f_hot
                total_combined += f_total
                fixture_rows.append({
                    "Fixture": name,
                    "Qty": qty,
                    "Cold WSFU": f"{f_cold:.1f}",
                    "Hot WSFU": f"{f_hot:.1f}",
                    "Total WSFU": f"{f_total:.1f}",
                })

        st.markdown("---")
        if fixture_rows:
            st.dataframe(pd.DataFrame(fixture_rows), width="stretch", hide_index=True)

        # Determine fixture type from selections
        has_flush_valve = False
        for name, _, _, _ in FIXTURE_TABLE:
            if "flush valve" in name.lower():
                qty = st.session_state.get(f"fix_{name}", 0)
                if qty > 0:
                    has_flush_valve = True
                    break
        detected_fixture_type = "Flush valve" if has_flush_valve else "Flush tank"

        # GPM demands
        cold_gpm = wsfu_to_gpm(total_cold, detected_fixture_type)
        hot_gpm = wsfu_to_gpm(total_hot, "Flush tank")  # hot always flush tank
        total_gpm = wsfu_to_gpm(total_combined, detected_fixture_type)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Cold WSFU", f"{total_cold:.1f}")
            st.metric("Cold GPM", f"{cold_gpm:.1f}")
        with m2:
            st.metric("Hot WSFU", f"{total_hot:.1f}")
            st.metric("Hot GPM", f"{hot_gpm:.1f}")
        with m3:
            st.metric("Total WSFU", f"{total_combined:.1f}")
            st.metric("Total GPM", f"{total_gpm:.1f}")

        if has_flush_valve:
            st.info("Flush valve fixtures detected — using flush valve demand curve for cold/total.")

        # Save to session state for pipe sizing page
        st.session_state["sys_cold_wsfu"] = total_cold
        st.session_state["sys_hot_wsfu"] = total_hot
        st.session_state["sys_total_wsfu"] = total_combined
        st.session_state["sys_fixture_type"] = detected_fixture_type

    # --- Pressure drop calculation ---
    with col_pressure:
        st.subheader("Pressure Budget")

        street_pressure = st.number_input(
            "Street pressure (psi)", min_value=1.0, max_value=200.0,
            value=60.0, step=1.0,
        )
        meter_loss = st.number_input(
            "Pressure drop through meter (psi)", min_value=0.0,
            max_value=50.0, value=5.0, step=0.5,
        )
        backflow_loss = st.number_input(
            "Pressure drop through backflow preventer (psi)",
            min_value=0.0, max_value=50.0, value=5.0, step=0.5,
        )
        height_ft = st.number_input(
            "Height — meter to highest fixture (ft)",
            min_value=0.0, max_value=500.0, value=10.0, step=1.0,
        )
        default_residual = 15.0 if has_flush_valve else 8.0
        residual_pressure = st.number_input(
            "Required residual pressure at fixture (psi)",
            min_value=0.0, max_value=50.0, value=default_residual, step=1.0,
        )

        st.markdown("---")
        st.subheader("Developed Length")
        developed_length = st.number_input(
            "Developed length of longest run (ft)",
            min_value=1.0, max_value=5000.0, value=100.0, step=5.0,
        )
        fitting_pct = st.number_input(
            "Fitting correction factor (%)",
            min_value=0, max_value=200, value=50, step=5,
            help="Added as a percentage of developed length to account for fittings. "
                 "e.g., 50% means total length = 1.5 × developed length.",
        )
        fitting_multiplier = 1.0 + fitting_pct / 100.0
        effective_length = developed_length * fitting_multiplier

        st.caption(
            f"Effective length: {developed_length:.0f} ft × {fitting_multiplier:.2f} "
            f"= **{effective_length:.0f} ft**"
        )

        # Pressure budget
        static_loss = height_ft * 0.433
        total_losses = meter_loss + backflow_loss + static_loss + residual_pressure
        available_pressure = street_pressure - total_losses

        st.markdown("---")
        st.text(
            f"  Street pressure:    {street_pressure:>7.1f} psi\n"
            f"− Meter loss:         {meter_loss:>7.1f} psi\n"
            f"− Backflow loss:      {backflow_loss:>7.1f} psi\n"
            f"− Static head:        {static_loss:>7.1f} psi  ({height_ft:.0f} ft × 0.433)\n"
            f"− Residual required:  {residual_pressure:>7.1f} psi\n"
            f"{'─' * 40}\n"
            f"= Available for friction: {available_pressure:>6.1f} psi\n"
            f"÷ Effective length:       {effective_length:>6.0f} ft\n"
        )

        if available_pressure <= 0:
            st.error("No pressure available for friction loss. "
                     "Reduce losses or increase supply pressure.")
            dp_calc = 0.0
        else:
            dp_calc = (available_pressure / effective_length) * 100.0
            st.success(f"**Uniform pressure drop: {dp_calc:.2f} psi/100ft**")

        # Save to session state
        st.session_state["sys_dp_limit"] = dp_calc


# ---------------------------------------------------------------------------
# Page: Pipe Sizing
# ---------------------------------------------------------------------------

def page_pipe_sizing():
    st.title("Pipe Sizing — UPC Appendix A")
    st.caption("Allowable flow per pipe size using Hazen-Williams")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Inputs")

        material = st.selectbox("Pipe material", list(HW_C.keys()))

        fixture_type = st.radio("Fixture type", ["Flush tank", "Flush valve"],
                                index=0 if st.session_state.get("sys_fixture_type", "Flush tank") == "Flush tank" else 1,
                                horizontal=True)

        water_type = st.radio("Water type", ["Cold", "Hot"], horizontal=True)

        hot_temp = None
        if water_type == "Hot":
            hot_temp = st.number_input(
                "Hot water temperature (°F)",
                min_value=100, max_value=210, value=120, step=5,
            )

        if water_type == "Cold":
            v_limit = 8.0
        elif hot_temp is not None and hot_temp >= 140:
            v_limit = 3.0
        else:
            v_limit = 5.0

        st.info(f"Velocity limit: **{v_limit} ft/s** "
                f"({'cold water' if water_type == 'Cold' else f'hot water @ {hot_temp}°F'})")

        # Pressure drop — use system value if available, allow override
        sys_dp = st.session_state.get("sys_dp_limit", None)
        if sys_dp and sys_dp > 0:
            st.caption(f"System-calculated pressure drop: {sys_dp:.2f} psi/100ft")
        dp_limit = st.number_input(
            "Allowable pressure drop (psi per 100 ft)",
            min_value=0.1, max_value=50.0,
            value=round(sys_dp, 1) if sys_dp and sys_dp > 0 else 4.0,
            step=0.5,
        )

        max_size = st.selectbox(
            "Maximum pipe size to display",
            ALL_SIZES,
            index=ALL_SIZES.index('6"') if material == "Copper (fairly rough)" else ALL_SIZES.index('2"'),
        )

        if material == "PEX":
            st.caption("PEX tubing is commonly available up to 2\".")

    # Build results
    if material == "Copper (fairly rough)":
        id_lookup = COPPER_TYPE_L_ID
    else:
        id_lookup = PEX_ID

    C = HW_C[material]
    max_size_in = SIZE_TO_INCHES[max_size]

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

    # --- Quick Lookup ---
    st.divider()
    st.subheader("Quick Lookup")

    wsfu_lookup_type = "Flush tank" if water_type == "Hot" else fixture_type

    def _find_min_pipe(gpm_needed):
        if not rows:
            return None
        for r in rows:
            if float(r["Allowed GPM"]) >= gpm_needed:
                return r["Nominal Size"]
        return None

    lookup_tab1, lookup_tab2 = st.tabs(["Lookup by GPM", "Lookup by Fixture Units"])

    with lookup_tab1:
        g1, g2, g3 = st.columns(3)
        with g1:
            lookup_gpm = st.number_input("Flow rate (GPM)", min_value=0.1, max_value=1000.0, value=10.0, step=1.0)
        with g2:
            st.metric("Equivalent WSFU", f"{gpm_to_wsfu(lookup_gpm, wsfu_lookup_type):.0f}")
        with g3:
            best = _find_min_pipe(lookup_gpm)
            st.metric("Minimum pipe size", best if best else "Exceeds range")

    with lookup_tab2:
        f1, f2, f3 = st.columns(3)
        with f1:
            lookup_wsfu = st.number_input("Fixture Units (WSFU)", min_value=1, max_value=10000, value=20, step=1)
        with f2:
            lookup_gpm_from_wsfu = wsfu_to_gpm(lookup_wsfu, wsfu_lookup_type)
            st.metric("Equivalent GPM", f"{lookup_gpm_from_wsfu:.1f}")
        with f3:
            best_fu = _find_min_pipe(lookup_gpm_from_wsfu)
            st.metric("Minimum pipe size", best_fu if best_fu else "Exceeds range")

    st.divider()
    st.caption(
        "Based on UPC Appendix A sizing methodology. "
        "Hazen-Williams C values: Copper (fairly rough) = 130, PEX = 150. "
        "Pipe IDs: Copper Type L per ASTM B88, PEX per ASTM F876 SDR-9. "
        "WSFU-to-GPM conversion per UPC Table A 4.1 (flush-tank and flush-valve systems). "
        "Velocity limits per UPC: 8 fps cold, 5 fps hot (<140°F), 3 fps hot (>=140°F)."
    )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

pg = st.navigation([
    st.Page(page_system_design, title="System Design", icon="\u2699\ufe0f"),
    st.Page(page_pipe_sizing, title="Pipe Sizing", icon="\U0001f4cf"),
])
pg.run()
