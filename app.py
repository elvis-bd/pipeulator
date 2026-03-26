import streamlit as st
import numpy as np
import pandas as pd
from datetime import date
from fpdf import FPDF

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
    "Copper (smooth)": 140,
    "Copper (fairly smooth)": 135,
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
# PDF Report Generation
# ---------------------------------------------------------------------------

class PipeulatorPDF(FPDF):
    def __init__(self, project_name="", engineer=""):
        super().__init__()
        self.project_name = project_name
        self.engineer = engineer

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 5, "Pipeulator - UPC Appendix A Pipe Sizing Report", align="C", new_x="LMARGIN", new_y="NEXT")
        if self.project_name:
            self.set_font("Helvetica", "", 8)
            self.cell(0, 4, f"Project: {self.project_name}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y() + 1, 200, self.get_y() + 1)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.cell(0, 10, f"Generated {date.today().isoformat()}"
                  f"{'  |  Engineer: ' + self.engineer if self.engineer else ''}"
                  f"  |  Page {self.page_no()}/{{nb}}",
                  align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 7, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def add_table(self, headers, data, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        # Header row
        self.set_font("Helvetica", "B", 7)
        self.set_fill_color(200, 200, 200)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 5, h, border=1, fill=True, align="C")
        self.ln()
        # Data rows
        self.set_font("Helvetica", "", 7)
        for row_idx, row in enumerate(data):
            fill = row_idx % 2 == 1
            if fill:
                self.set_fill_color(245, 245, 245)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 5, str(val), border=1, fill=fill, align="C")
            self.ln()

    def add_kv(self, label, value, indent=4):
        self.set_font("Helvetica", "", 8)
        self.cell(indent)
        self.cell(75, 5, label)
        self.set_font("Helvetica", "B", 8)
        self.cell(0, 5, str(value), new_x="LMARGIN", new_y="NEXT")


def generate_system_pdf(project_name, engineer, fixture_rows, totals, pressure_data, other_rows=None):
    """Generate PDF for System Design page."""
    pdf = PipeulatorPDF(project_name, engineer)
    pdf.alias_nb_pages()
    pdf.add_page()

    # --- Fixture Count ---
    pdf.section_title("Fixture Count (per UPC Table 6-4)")
    if fixture_rows:
        headers = ["Fixture", "Qty", "Cold WSFU", "Hot WSFU", "Total WSFU"]
        widths = [70, 20, 30, 30, 30]
        data = [[r["Fixture"], r["Qty"], r["Cold WSFU"], r["Hot WSFU"], r["Total WSFU"]]
                for r in fixture_rows]
        pdf.add_table(headers, data, widths)
    else:
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 5, "  No fixtures entered.", new_x="LMARGIN", new_y="NEXT")

    if other_rows:
        pdf.ln(2)
        pdf.section_title("Other Fixtures / Devices (direct GPM)")
        headers = ["Description", "GPM", "Water Type"]
        widths = [100, 40, 40]
        data = [[r["Fixture"], r["GPM"], r["Water"]] for r in other_rows]
        pdf.add_table(headers, data, widths)
    pdf.ln(3)

    # --- Demand Summary ---
    pdf.section_title("System Demand Summary")
    pdf.add_kv("Cold WSFU:", f"{totals['cold_wsfu']:.1f}")
    pdf.add_kv("Cold GPM:", f"{totals['cold_gpm']:.1f}")
    pdf.add_kv("Hot WSFU:", f"{totals['hot_wsfu']:.1f}")
    pdf.add_kv("Hot GPM:", f"{totals['hot_gpm']:.1f}")
    pdf.add_kv("Total WSFU:", f"{totals['total_wsfu']:.1f}")
    pdf.add_kv("Total GPM:", f"{totals['total_gpm']:.1f}")
    pdf.add_kv("Fixture Type:", totals['fixture_type'])
    pdf.ln(3)

    # --- Pressure Budget ---
    _render_pressure_budget_pdf(pdf, pressure_data)

    return bytes(pdf.output())


def _render_pressure_budget_pdf(pdf, p):
    """Render pressure budget section to PDF, handling booster pump case."""
    pdf.section_title("Pressure Budget")
    if p.get("use_booster"):
        pdf.add_kv("Supply:", "Booster pump")
        pdf.add_kv("Booster discharge pressure:", f"{p['booster_pressure']:.1f} psi")
    else:
        pdf.add_kv("Street pressure:", f"{p['street_pressure']:.1f} psi")
        pdf.add_kv("Meter loss:", f"{p['meter_loss']:.1f} psi")
        pdf.add_kv("Backflow preventer loss:", f"{p['backflow_loss']:.1f} psi")
    pdf.add_kv("Height to highest fixture:", f"{p['height_ft']:.0f} ft")
    pdf.add_kv("Static head loss:", f"{p['static_loss']:.1f} psi  ({p['height_ft']:.0f} ft x 0.433)")
    pdf.add_kv("Residual pressure required:", f"{p['residual_pressure']:.1f} psi")
    pdf.add_kv("Available for friction:", f"{p['available_pressure']:.1f} psi")
    pdf.ln(2)

    pdf.section_title("Developed Length & Fitting Factor")
    pdf.add_kv("Developed length:", f"{p['developed_length']:.0f} ft")
    pdf.add_kv("Fitting correction:", f"{p['fitting_pct']}%  (x{p['fitting_multiplier']:.2f})")
    pdf.add_kv("Effective length:", f"{p['effective_length']:.0f} ft")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(0, 100, 0)
    pdf.cell(0, 7, f"  Uniform Pressure Drop: {p['dp_calc']:.2f} psi/100ft",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)


def generate_sizing_pdf(project_name, engineer, params, sizing_rows):
    """Generate PDF for Pipe Sizing page."""
    pdf = PipeulatorPDF(project_name, engineer)
    pdf.alias_nb_pages()
    pdf.add_page()

    # --- Parameters ---
    pdf.section_title("Sizing Parameters")
    pdf.add_kv("Pipe material:", params["material"])
    pdf.add_kv("Hazen-Williams C:", str(params["C"]))
    pdf.add_kv("Fixture type:", params["fixture_type"])
    pdf.add_kv("Water type:", params["water_type"])
    if params["water_type"] == "Hot" and params.get("hot_temp"):
        pdf.add_kv("Hot water temperature:", f"{params['hot_temp']} F")
    pdf.add_kv("Velocity limit:", f"{params['v_limit']:.0f} ft/s")
    pdf.add_kv("Pressure drop:", f"{params['dp_limit']:.2f} psi/100ft")
    pdf.ln(3)

    # --- Sizing Table ---
    pdf.section_title("Pipe Sizing Table")
    if sizing_rows:
        headers = ["Size", "ID (in)", "GPM (press.)", "GPM (vel.)",
                   "Allowed GPM", "Vel. (ft/s)", "WSFU", "Limiting"]
        widths = [18, 18, 24, 24, 24, 22, 22, 30]
        data = [[r["Nominal Size"], r["ID (in)"], r["Max GPM (pressure)"],
                 r["Max GPM (velocity)"], r["Allowed GPM"], r["Velocity (ft/s)"],
                 r["Fixture Units (WSFU)"], r["Limiting Factor"]]
                for r in sizing_rows]
        pdf.add_table(headers, data, widths)
    else:
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 5, "  No pipe sizes available.", new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


def generate_full_pdf(project_name, engineer, fixture_rows, totals,
                      pressure_data, params, sizing_rows, other_rows=None):
    """Generate combined PDF with both system design and pipe sizing."""
    pdf = PipeulatorPDF(project_name, engineer)
    pdf.alias_nb_pages()
    pdf.add_page()

    # --- Fixture Count ---
    pdf.section_title("Fixture Count (per UPC Table 6-4)")
    if fixture_rows:
        headers = ["Fixture", "Qty", "Cold WSFU", "Hot WSFU", "Total WSFU"]
        widths = [70, 20, 30, 30, 30]
        data = [[r["Fixture"], r["Qty"], r["Cold WSFU"], r["Hot WSFU"], r["Total WSFU"]]
                for r in fixture_rows]
        pdf.add_table(headers, data, widths)
    else:
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 5, "  No fixtures entered.", new_x="LMARGIN", new_y="NEXT")

    if other_rows:
        pdf.ln(2)
        pdf.section_title("Other Fixtures / Devices (direct GPM)")
        headers = ["Description", "GPM", "Water Type"]
        widths = [100, 40, 40]
        data = [[r["Fixture"], r["GPM"], r["Water"]] for r in other_rows]
        pdf.add_table(headers, data, widths)
    pdf.ln(3)

    # --- Demand Summary ---
    pdf.section_title("System Demand Summary")
    pdf.add_kv("Cold WSFU:", f"{totals['cold_wsfu']:.1f}")
    pdf.add_kv("Cold GPM:", f"{totals['cold_gpm']:.1f}")
    pdf.add_kv("Hot WSFU:", f"{totals['hot_wsfu']:.1f}")
    pdf.add_kv("Hot GPM:", f"{totals['hot_gpm']:.1f}")
    pdf.add_kv("Total WSFU:", f"{totals['total_wsfu']:.1f}")
    pdf.add_kv("Total GPM:", f"{totals['total_gpm']:.1f}")
    pdf.add_kv("Fixture Type:", totals['fixture_type'])
    pdf.ln(3)

    # --- Pressure Budget ---
    _render_pressure_budget_pdf(pdf, pressure_data)
    pdf.ln(3)

    # --- Pipe Sizing ---
    pdf.add_page()
    pdf.section_title("Sizing Parameters")
    pdf.add_kv("Pipe material:", params["material"])
    pdf.add_kv("Hazen-Williams C:", str(params["C"]))
    pdf.add_kv("Fixture type:", params["fixture_type"])
    pdf.add_kv("Water type:", params["water_type"])
    if params["water_type"] == "Hot" and params.get("hot_temp"):
        pdf.add_kv("Hot water temperature:", f"{params['hot_temp']} F")
    pdf.add_kv("Velocity limit:", f"{params['v_limit']:.0f} ft/s")
    pdf.add_kv("Pressure drop:", f"{params['dp_limit']:.2f} psi/100ft")
    pdf.ln(3)

    pdf.section_title("Pipe Sizing Table")
    if sizing_rows:
        headers = ["Size", "ID (in)", "GPM (press.)", "GPM (vel.)",
                   "Allowed GPM", "Vel. (ft/s)", "WSFU", "Limiting"]
        widths = [18, 18, 24, 24, 24, 22, 22, 30]
        data = [[r["Nominal Size"], r["ID (in)"], r["Max GPM (pressure)"],
                 r["Max GPM (velocity)"], r["Allowed GPM"], r["Velocity (ft/s)"],
                 r["Fixture Units (WSFU)"], r["Limiting Factor"]]
                for r in sizing_rows]
        pdf.add_table(headers, data, widths)

    return bytes(pdf.output())


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

        # --- Other / custom fixtures ---
        st.markdown("---")
        st.markdown("**Other fixtures / devices**")

        if "other_count" not in st.session_state:
            st.session_state["other_count"] = 0

        # Render input rows for each custom fixture
        other_rows = []
        items_to_remove = []
        for i in range(st.session_state["other_count"]):
            oc1, oc2, oc3, oc4 = st.columns([3, 1, 2, 1])
            with oc1:
                desc = st.text_input("Description", value="", key=f"other_desc_{i}",
                                     placeholder="e.g., Cooling tower makeup",
                                     label_visibility="collapsed" if i > 0 else "visible")
            with oc2:
                gpm = st.number_input("GPM", min_value=0.0, max_value=5000.0,
                                      value=0.0, step=1.0, key=f"other_gpm_{i}",
                                      label_visibility="collapsed" if i > 0 else "visible")
            with oc3:
                water = st.radio("Type", ["Cold", "Hot", "Both"],
                                 horizontal=True, key=f"other_water_{i}",
                                 label_visibility="collapsed" if i > 0 else "visible")
            with oc4:
                if i > 0:
                    st.markdown("<br>", unsafe_allow_html=True)
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                if st.button("X", key=f"remove_other_{i}"):
                    items_to_remove.append(i)

            if desc and gpm > 0:
                other_rows.append({
                    "Fixture": desc,
                    "Qty": 1,
                    "GPM": f"{gpm:.1f}",
                    "Water": water,
                })

        # Handle removals
        if items_to_remove:
            # Shift session state keys down to fill gaps
            old_count = st.session_state["other_count"]
            keep = [j for j in range(old_count) if j not in items_to_remove]
            for new_i, old_i in enumerate(keep):
                if new_i != old_i:
                    for suffix in ("desc", "gpm", "water"):
                        key_old = f"other_{suffix}_{old_i}"
                        key_new = f"other_{suffix}_{new_i}"
                        if key_old in st.session_state:
                            st.session_state[key_new] = st.session_state[key_old]
            # Clean up trailing keys
            for j in range(len(keep), old_count):
                for suffix in ("desc", "gpm", "water"):
                    st.session_state.pop(f"other_{suffix}_{j}", None)
            st.session_state["other_count"] = len(keep)
            st.rerun()

        if st.button("Add fixture"):
            st.session_state["other_count"] += 1
            st.rerun()

        st.markdown("---")
        if fixture_rows:
            st.dataframe(pd.DataFrame(fixture_rows), width="stretch", hide_index=True)
        if other_rows:
            st.dataframe(pd.DataFrame(other_rows), width="stretch", hide_index=True)

        # Determine fixture type from selections
        has_flush_valve = False
        for name, _, _, _ in FIXTURE_TABLE:
            if "flush valve" in name.lower():
                qty = st.session_state.get(f"fix_{name}", 0)
                if qty > 0:
                    has_flush_valve = True
                    break
        detected_fixture_type = "Flush valve" if has_flush_valve else "Flush tank"

        # GPM demands from WSFU fixtures
        cold_gpm = wsfu_to_gpm(total_cold, detected_fixture_type)
        hot_gpm = wsfu_to_gpm(total_hot, "Flush tank")  # hot always flush tank
        total_gpm = wsfu_to_gpm(total_combined, detected_fixture_type)

        # Add other GPM demands directly
        for i in range(st.session_state.get("other_count", 0)):
            o_gpm = st.session_state.get(f"other_gpm_{i}", 0.0)
            o_water = st.session_state.get(f"other_water_{i}", "Cold")
            if o_gpm > 0:
                if o_water == "Cold":
                    cold_gpm += o_gpm
                    total_gpm += o_gpm
                elif o_water == "Hot":
                    hot_gpm += o_gpm
                    total_gpm += o_gpm
                else:  # Both
                    cold_gpm += o_gpm
                    hot_gpm += o_gpm
                    total_gpm += o_gpm

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

        use_booster = st.toggle("Booster pump", value=False)

        if use_booster:
            booster_pressure = st.number_input(
                "Booster pump discharge pressure (psi)",
                min_value=1.0, max_value=300.0, value=80.0, step=1.0,
            )
            starting_pressure = booster_pressure
            street_pressure = 0.0
            meter_loss = 0.0
            backflow_loss = 0.0
        else:
            booster_pressure = 0.0
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
            starting_pressure = street_pressure

        height_ft = st.number_input(
            "Height to highest fixture (ft)",
            min_value=0.0, max_value=500.0, value=10.0, step=1.0,
        )
        default_residual = 25.0
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
                 "e.g., 50% means total length = 1.5 x developed length.",
        )
        fitting_multiplier = 1.0 + fitting_pct / 100.0
        effective_length = developed_length * fitting_multiplier

        st.caption(
            f"Effective length: {developed_length:.0f} ft x {fitting_multiplier:.2f} "
            f"= **{effective_length:.0f} ft**"
        )

        # Pressure budget
        static_loss = height_ft * 0.433
        if use_booster:
            total_losses = static_loss + residual_pressure
            available_pressure = booster_pressure - total_losses
        else:
            total_losses = meter_loss + backflow_loss + static_loss + residual_pressure
            available_pressure = street_pressure - total_losses

        st.markdown("---")
        if use_booster:
            st.text(
                f"  Booster pressure:   {booster_pressure:>7.1f} psi\n"
                f"- Static head:        {static_loss:>7.1f} psi  ({height_ft:.0f} ft x 0.433)\n"
                f"- Residual required:  {residual_pressure:>7.1f} psi\n"
                f"{'=' * 40}\n"
                f"= Available for friction: {available_pressure:>6.1f} psi\n"
                f"/ Effective length:       {effective_length:>6.0f} ft\n"
            )
        else:
            st.text(
                f"  Street pressure:    {street_pressure:>7.1f} psi\n"
                f"- Meter loss:         {meter_loss:>7.1f} psi\n"
                f"- Backflow loss:      {backflow_loss:>7.1f} psi\n"
                f"- Static head:        {static_loss:>7.1f} psi  ({height_ft:.0f} ft x 0.433)\n"
                f"- Residual required:  {residual_pressure:>7.1f} psi\n"
                f"{'=' * 40}\n"
                f"= Available for friction: {available_pressure:>6.1f} psi\n"
                f"/ Effective length:       {effective_length:>6.0f} ft\n"
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

    # --- Save data for PDF export ---
    totals = {
        "cold_wsfu": total_cold, "hot_wsfu": total_hot, "total_wsfu": total_combined,
        "cold_gpm": cold_gpm, "hot_gpm": hot_gpm, "total_gpm": total_gpm,
        "fixture_type": detected_fixture_type,
    }
    pressure_data = {
        "use_booster": use_booster, "booster_pressure": booster_pressure,
        "street_pressure": street_pressure, "meter_loss": meter_loss,
        "backflow_loss": backflow_loss, "height_ft": height_ft,
        "static_loss": static_loss, "residual_pressure": residual_pressure,
        "available_pressure": available_pressure,
        "developed_length": developed_length, "fitting_pct": fitting_pct,
        "fitting_multiplier": fitting_multiplier,
        "effective_length": effective_length, "dp_calc": dp_calc,
    }
    st.session_state["sys_fixture_rows"] = fixture_rows
    st.session_state["sys_other_rows"] = other_rows
    st.session_state["sys_totals"] = totals
    st.session_state["sys_pressure_data"] = pressure_data

    # --- PDF Download ---
    st.divider()
    st.subheader("Export to PDF")
    pc1, pc2 = st.columns(2)
    with pc1:
        project_name = st.text_input("Project name", value=st.session_state.get("pdf_project", ""),
                                     key="sys_project_name")
        st.session_state["pdf_project"] = project_name
    with pc2:
        engineer = st.text_input("Engineer", value=st.session_state.get("pdf_engineer", ""),
                                 key="sys_engineer")
        st.session_state["pdf_engineer"] = engineer

    pdf_bytes = generate_system_pdf(project_name, engineer, fixture_rows, totals, pressure_data, other_rows)
    st.download_button(
        "Download System Design PDF",
        data=pdf_bytes,
        file_name="pipeulator_system_design.pdf",
        mime="application/pdf",
    )


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
            index=ALL_SIZES.index('6"') if material.startswith("Copper") else ALL_SIZES.index('2"'),
        )

        if material == "PEX":
            st.caption("PEX tubing is commonly available up to 2\".")

    # Build results
    if material.startswith("Copper"):
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

    # --- PDF Export ---
    st.divider()
    st.subheader("Export to PDF")

    params = {
        "material": material, "C": C, "fixture_type": fixture_type,
        "water_type": water_type, "hot_temp": hot_temp,
        "v_limit": v_limit, "dp_limit": dp_limit,
    }

    pc1, pc2 = st.columns(2)
    with pc1:
        project_name = st.text_input("Project name", value=st.session_state.get("pdf_project", ""),
                                     key="sizing_project_name")
        st.session_state["pdf_project"] = project_name
    with pc2:
        engineer = st.text_input("Engineer", value=st.session_state.get("pdf_engineer", ""),
                                 key="sizing_engineer")
        st.session_state["pdf_engineer"] = engineer

    dl1, dl2 = st.columns(2)
    with dl1:
        sizing_pdf = generate_sizing_pdf(project_name, engineer, params, rows)
        st.download_button(
            "Download Pipe Sizing PDF",
            data=sizing_pdf,
            file_name="pipeulator_pipe_sizing.pdf",
            mime="application/pdf",
        )
    with dl2:
        # Full report only if system design data exists
        sys_fixture_rows = st.session_state.get("sys_fixture_rows")
        sys_totals = st.session_state.get("sys_totals")
        sys_pressure = st.session_state.get("sys_pressure_data")
        if sys_totals and sys_pressure:
            sys_other_rows = st.session_state.get("sys_other_rows", [])
            full_pdf = generate_full_pdf(
                project_name, engineer, sys_fixture_rows, sys_totals,
                sys_pressure, params, rows, sys_other_rows,
            )
            st.download_button(
                "Download Full Report PDF",
                data=full_pdf,
                file_name="pipeulator_full_report.pdf",
                mime="application/pdf",
            )
        else:
            st.caption("Complete System Design page first for full report.")

    st.divider()
    st.caption(
        "Based on UPC Appendix A sizing methodology. "
        "Hazen-Williams C values: Copper smooth=140, fairly smooth=135, fairly rough=130; PEX=150. "
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
