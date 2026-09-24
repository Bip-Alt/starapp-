# AppStar.  Run with:  streamlit run star_app.py
#
# Python runs once per metallicity: it precomputes the star's state over
# a grid of masses and ages, then ships one Altair chart whose mass and
# age sliders are Vega parameters. Dragging them filters the grid in the
# browser, so the star updates live, with no Python rerun. Releasing the
# metallicity slider (a Streamlit widget) reruns the script and rebuilds
# the grid at the new Z.
#
# ======================================================================
#Updated code as required in the exercise  (Part D exercise)
# ----------------------------------------------------------------------
# I have catptured this as  "# CHANGED" and "# NEW" to find every edit.
#
#   1. Metallicity slider
#      - The constant Z = 0.02 is replaced by a Streamlit slider (lz).
#      - A caption shows Z, the neutron star / black hole boundary,
#        and whether the pair-instability window is open.
#
#   2. Pair-instability branch
#      - star_state() gains a "no remnant" phase for stars of
#        140-260 suns when Z < 0.001.
#      - The grid loop draws that phase black on black (nothing left).
#
#   3. Hertzsprung-Russell diagram replaces the phase plane
#      - The whole right-hand phase-plane section is removed.
#      - New HR diagram: temperature (reversed, log) against
#        luminosity (log, 1e-4 to 1e6 suns).
#      - Layers: main-sequence band, white-dwarf cooling track,
#        spectral-class rules and letters, region labels, the Sun,
#        and your star riding the mass and age sliders.
#
#   5. Bug fix after testing
#      - Default age slider 0.66 -> 0.68 so the star appears on load
#        and after each metallicity rerun.
#      - Dashed solar-metallicity reference line on the HR diagram so
#        the metallicity shift is visible.
#
#   4. Housekeeping
#      - st.altair_chart: deprecated use_container_width=False is
#        replaced by width="content".
# ======================================================================

import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

SUN_T = 5772.0

st.set_page_config(page_title="AppStar", layout="wide")

st.markdown("""<style>
.stApp {background-color: #000000;}
.stApp, .stApp p, .stApp label {color: #e8e8e8;}
.block-container {padding-top: 0.4rem; padding-bottom: 0.3rem;
                  max-width: 1000px;}
header[data-testid="stHeader"] {display: none;}
h1, h2, h3 {padding-top: 0 !important; margin: 0 0 0.2rem !important;
            color: #f0f0f0;}
form.vega-bindings {display: flex; justify-content: center;
                    gap: 3rem; margin-top: 0.6rem;
                    color: #e8e8e8; font-weight: 700;}
form.vega-bindings input[type="range"] {width: 240px;}
</style>""", unsafe_allow_html=True)

st.markdown("### AppStar")

alt.data_transformers.disable_max_rows()

# ---- CHANGED (1): metallicity is now a Streamlit slider --------------
# Was:   Z = 0.02          # metallicity; extension: make this a slider
lz = st.slider("log10 metallicity", -4.0, -1.4, -1.7, step=0.05)
Z = 10 ** lz
zr = Z / 0.02

# ---- NEW (1, 2): pair-instability limits and a status caption --------
PAIR_LO, PAIR_HI = 140.0, 260.0
bh_boundary = 18 + 7 * zr
st.caption(
    f"Z = {Z:.5f} ({zr:.2f} × solar)  |  "
    f"neutron star / black hole boundary: {bh_boundary:.1f} suns  |  "
    f"pair-instability window ({PAIR_LO:.0f}–{PAIR_HI:.0f} suns): "
    f"{'open' if Z < 0.001 else 'closed'}")


def bb_rgb(T):
    # Approximate black-body colour, valid from about 1000 to 40000 K.
    t = T / 100.0
    r = 255.0 if t <= 66 else 329.7 * (t - 60) ** -0.1332
    g = 99.47 * math.log(t) - 161.1 if t <= 66 else 288.1 * (t - 60) ** -0.0755
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.5 * math.log(t - 10) - 305.0
    return tuple(min(255.0, max(0.0, v)) / 255 for v in (r, g, b))


def rgb_str(T):
    r, g, b = (int(round(255 * c)) for c in bb_rgb(min(T, 40000)))
    return f"rgb({r},{g},{b})"


def star_state(mass, age):
    # the same rules as the course page
    L = mass ** 3.5 * zr ** -0.1        # luminosity, suns  [uses zr]
    R = mass ** 0.8                     # radius, suns
    T = SUN_T * (L / R ** 2) ** 0.25    # surface temperature, K
    t_pre = 0.03 * mass ** -1.5
    t_ms = (10.0 * mass ** -2.5 * (1 + 2.5 * math.exp(-mass / 0.12))
            + 0.0025)
    t_g = 1.15 * t_ms

    if age <= t_pre:
        phase = "protostar"
    elif age <= t_ms:
        phase = "main sequence"
    elif mass < 0.25:
        phase = "white dwarf"           # fully convective: no giant
    elif mass < 8:
        phase = "red giant" if age <= t_g else "white dwarf"
    elif age <= t_g:
        frac = (age - t_ms) / (t_g - t_ms)
        phase = "blue supergiant" if frac < 0.4 else "red supergiant"
    elif age <= 1.10 * t_g:
        phase = "supernova"
    # ---- NEW (2): pair instability, the star blows itself apart ------
    elif Z < 0.001 and PAIR_LO <= mass <= PAIR_HI:
        phase = "no remnant"
    elif mass < 18 + 7 * zr:            # remnant boundary  [uses zr]
        phase = "neutron star"
    else:
        phase = "black hole"

    T_show, L_show, R_show = float(T), L, R
    if phase == "protostar":
        T_show, L_show, R_show = 0.75 * T, 2 * L, 3 * R
    elif phase == "red giant":
        T_show = 3900.0
        R_show = max(R * 60, 10.0)
        L_show = R_show ** 2 * (T_show / SUN_T) ** 4
    elif phase == "blue supergiant":
        frac = (age - t_ms) / (t_g - t_ms)
        T_show = 12000.0
        R_show = 30 + 120 * frac
        L_show = R_show ** 2 * (T_show / SUN_T) ** 4
    elif phase == "red supergiant":
        frac = (age - t_ms) / (t_g - t_ms)
        T_show = 3500.0
        R_show = min(200 + 900 * frac, 900)
        L_show = R_show ** 2 * (T_show / SUN_T) ** 4
    elif phase == "white dwarf":
        cool = max(age - (t_ms if mass < 0.25 else t_g), 0.001)
        T_show = float(np.clip(60000.0 * (0.01 / cool) ** 0.3,
                               3500, 150000))
        R_show = 0.009
        L_show = R_show ** 2 * (T_show / SUN_T) ** 4
    elif phase == "supernova":
        # no luminosity: an explosion is an event, not an equilibrium state,
        # and its ~5e9 suns would stretch an HR luminosity axis by four
        # decades to hold one transient point
        T_show, L_show, R_show = 8000.0, None, None
    # ---- NEW (2): nothing is left, so no temperature, light or size --
    elif phase == "no remnant":
        T_show, L_show, R_show = None, None, None
    elif phase == "neutron star":
        T_show, L_show, R_show = 1e6, None, 1.7e-5
    elif phase == "black hole":
        T_show, L_show, R_show = None, None, 4.2e-6 * mass / 10
    return phase, T_show, L_show, R_show, t_ms


# ---- the grid: one row per slider combination ------------------------
lms = [round(-1.0 + 0.05 * k, 2) for k in range(70)]   # mass 0.1 to 282
las = [round(-4.0 + 0.06 * k, 2) for k in range(127)]  # age 1e-4 to 3631

rows = []
for lm in lms:
    for la in las:
        m = 10.0 ** lm
        a = 10.0 ** la
        phase, T, L, R, t_ms = star_state(m, a)
        if phase == "black hole":
            colour, px = "rgb(16,16,16)", 40.0
        elif phase == "neutron star":
            colour, px = "#CDE7FF", 6.0
        elif phase == "supernova":
            colour, px = "#FFD27D", 150.0
        # ---- NEW (2): draw "no remnant" black on black ---------------
        elif phase == "no remnant":
            colour, px = "rgb(0,0,0)", 5.0
        else:
            colour = rgb_str(T)
            px = float(np.clip(14 + 26 * (np.log10(R) + 2.2), 5, 150))
        rows.append(dict(
            lm=lm, la=la, mass=m, age=a,
            temp_K=T, lum=L, rad=R,
            colour=colour, size=px ** 2, phase=phase,
            massage=f"mass {m:.2g} suns, age {a:.2g} Gyr",
            temp=f"surface {T:,.0f} K" if T else "",
            lr=(f"luminosity {L:.3g} suns, radius {R:.3g} suns"
                if L and R else
                f"radius {R:.3g} suns" if R else ""),
            life=f"main-sequence lifetime {t_ms:.2g} Gyr",
        ))
grid = pd.DataFrame(rows)

m_sel = alt.param(name="m_sel", value=0.0, bind=alt.binding_range(
    min=-1.0, max=2.45, step=0.05, name="log10 mass (suns)  "))
# ---- CHANGED (5): default age 0.66 -> 0.68 ---------------------------
# The age grid runs -4.0, -3.94, ... so it holds 0.62 and 0.68 but not
# 0.66. With 0.66 the pick filter matched no row, so the star vanished
# from both panels on load and after every metallicity rerun (a rerun
# resets the mass and age sliders to their defaults).
a_sel = alt.param(name="a_sel", value=0.68, bind=alt.binding_range(
    min=-4.0, max=3.56, step=0.06, name="log10 age (Gyr)  "))
pick = ("abs(datum.lm - m_sel) < 0.02"
        " && abs(datum.la - a_sel) < 0.02")

# ---- the portrait (unchanged) ----------------------------------------
CX, CY = 160, 168
disc = alt.Chart(grid).transform_filter(pick).mark_circle(
    opacity=1).encode(
    x=alt.value(CX), y=alt.value(CY),
    size=alt.Size("size:Q", scale=None, legend=None),
    color=alt.Color("colour:N", scale=None, legend=None))
bh_ring = alt.Chart(grid).transform_filter(
    pick + ' && datum.phase == "black hole"').mark_point(
    filled=False, size=2400, stroke="#E07000", strokeWidth=3,
    opacity=1).encode(x=alt.value(CX), y=alt.value(CY))
# the Sun's size on the same log scale, for reference
sun_ring = alt.Chart(pd.DataFrame({"z": [0]})).mark_point(
    filled=False, size=int((14 + 26 * 2.2) ** 2),
    stroke="#DAA520", strokeWidth=1.5, opacity=1).encode(
    x=alt.value(CX), y=alt.value(CY))


def readout(field, y_px, size=12, color="#9aa1a8", bold=False):
    return alt.Chart(grid).transform_filter(pick).mark_text(
        fontSize=size, color=color,
        fontWeight="bold" if bold else "normal").encode(
        x=alt.value(CX), y=alt.value(y_px), text=field)


portrait = alt.layer(
    disc, bh_ring, sun_ring,
    readout("massage:N", 340),
    readout("phase:N", 364, size=15, color="#f5f2ea", bold=True),
    readout("temp:N", 386),
    readout("lr:N", 404),
    readout("life:N", 422),
).properties(width=320, height=440)

# ======================================================================
# CHANGED (3): the whole phase-plane section that was here is REMOVED
# and replaced by the Hertzsprung-Russell diagram below.
# (Removed: Ml/pre_l/ms_l/g_l arrays, region(), stage areas, supernova
#  line, age-of-universe rule, plane labels, class labels, Sun marker,
#  the "you" star on mass-age axes, and the `plane` layer.)
# ======================================================================

# ---- NEW (3): axes for the HR diagram --------------------------------
L_MIN, L_MAX = 1e-4, 1e6
T_MIN, T_MAX = 1500, 200000

T_SCALE = alt.Scale(type="log", domain=[T_MIN, T_MAX], reverse=True,
                    nice=False, clamp=True)      # hot on the left
L_SCALE = alt.Scale(type="log", domain=[L_MIN, L_MAX], nice=False,
                    clamp=True)
AXIS_STYLE = dict(gridColor="#2b303b", labelColor="#c8c8c8",
                  titleColor="#c8c8c8")
HX = alt.X("temp_K:Q", title="surface temperature (K), hot on the left",
           scale=T_SCALE,
           axis=alt.Axis(values=[2000, 5000, 10000, 20000, 50000, 100000],
                         format="~s", **AXIS_STYLE))
HY = alt.Y("lum:Q", title="luminosity (suns), log scale",
           scale=L_SCALE,
           axis=alt.Axis(values=[1e-4, 1e-2, 1, 1e2, 1e4, 1e6],
                         format="~g", **AXIS_STYLE))

# ---- NEW (3): main-sequence band (static layer) ----------------------
# every mass at its own temperature and luminosity, using the same
# metallicity-aware rules as star_state so the star sits exactly on the
# band while it is on the main sequence
Ml = np.geomspace(0.1, 300, 200)
ms_L = Ml ** 3.5 * zr ** -0.1
ms_T = SUN_T * (ms_L / (Ml ** 0.8) ** 2) ** 0.25
ms_df = pd.DataFrame({"mass": Ml, "temp_K": ms_T, "lum": ms_L})
ms_df = ms_df[(ms_df["lum"] >= L_MIN) & (ms_df["lum"] <= L_MAX)]
ms_band = alt.Chart(ms_df).mark_line(
    strokeWidth=14, color="#3a4150", opacity=0.9,
    strokeCap="round").encode(x=HX, y=HY, order="mass:Q")

# ---- NEW (5): solar-metallicity reference line -----------------------
# Metallicity only scales luminosity by zr ** -0.1, a shift too small to
# see on ten decades without something to compare against. This dashed
# line is the main sequence at Z = 0.02; the thick band moves away from
# it as the metallicity slider moves.
ref_L = Ml ** 3.5
ref_T = SUN_T * (ref_L / (Ml ** 0.8) ** 2) ** 0.25
ref_df = pd.DataFrame({"mass": Ml, "temp_K": ref_T, "lum": ref_L})
ref_df = ref_df[(ref_df["lum"] >= L_MIN) & (ref_df["lum"] <= L_MAX)]
ms_ref = alt.Chart(ref_df).mark_line(
    strokeWidth=1.5, color="#DAA520", strokeDash=[4, 4],
    opacity=0.8).encode(x=HX, y=HY, order="mass:Q")
ref_label = alt.Chart(pd.DataFrame(
    {"temp_K": [30000], "lum": [3e5]})).mark_text(
    align="left", dx=8, fontSize=10, color="#DAA520").encode(
    x=HX, y=HY, text=alt.value("solar metallicity"))

# ---- NEW (3): white-dwarf cooling track (optional context) -----------
wd_T = np.geomspace(3500, 150000, 80)
wd_df = pd.DataFrame({"temp_K": wd_T,
                      "lum": 0.009 ** 2 * (wd_T / SUN_T) ** 4})
wd_df = wd_df[wd_df["lum"] >= L_MIN]
wd_track = alt.Chart(wd_df).mark_line(
    strokeWidth=2, color="#b7aec4", strokeDash=[5, 4],
    opacity=0.7).encode(x=HX, y=HY)

# ---- NEW (3): spectral-class boundaries and letters (optional) -------
cls_edges = pd.DataFrame({"temp_K": [3700, 5200, 6000, 7500, 10000, 30000]})
cls_rules = alt.Chart(cls_edges).mark_rule(
    color="#2b303b", strokeDash=[2, 3]).encode(x=HX)
cls_letters = alt.Chart(pd.DataFrame({
    "temp_K": [3050, 4400, 5580, 6700, 8700, 17300, 45000],
    "lum": [3e5] * 7,
    "t": list("MKGFABO"),
})).mark_text(fontSize=11, fontWeight=600, color="#8a8f98").encode(
    x=HX, y=HY, text="t:N")

# ---- NEW (3): region labels (optional) -------------------------------
hr_labels = alt.Chart(pd.DataFrame({
    "temp_K": [9000, 4300, 5500, 22000],
    "lum":    [2e3, 150, 3e4, 4e-4],
    "t":      ["main sequence", "giants", "supergiants", "white dwarfs"],
    "c":      ["#8a8f98", "#c73b25", "#e0a060", "#b7aec4"],
})).mark_text(fontSize=11).encode(
    x=HX, y=HY, text="t:N",
    color=alt.Color("c:N", scale=None, legend=None))

# ---- NEW (3): the Sun, for reference ---------------------------------
sun_df = pd.DataFrame({"temp_K": [SUN_T], "lum": [1.0]})
sun_pt = alt.Chart(sun_df).mark_circle(
    size=55, color="#1e7d32", opacity=1).encode(x=HX, y=HY)
sun_txt = alt.Chart(sun_df).mark_text(
    dy=14, fontSize=11, fontWeight=700, color="#1e7d32").encode(
    x=HX, y=HY, text=alt.value("Sun"))

# ---- CHANGED (3): your star now rides temp_K / lum, not mass / age ---
# rows with no luminosity (supernova, no remnant, neutron star,
# black hole) are filtered out and simply vanish from the diagram
you = alt.Chart(grid).transform_filter(
    pick + " && isValid(datum.lum) && isValid(datum.temp_K)"
).mark_point(
    shape=("M 0 -1 L 0.24 -0.31 L 0.95 -0.31 L 0.38 0.12 L 0.59 0.81"
           " L 0 0.38 L -0.59 0.81 L -0.38 0.12 L -0.95 -0.31"
           " L -0.24 -0.31 Z"),
    filled=True, size=280, color="#FFC300",
    stroke="#8C6A2F", strokeWidth=1.2, opacity=1).encode(x=HX, y=HY)

# ---- NEW (3): the HR diagram layer, replacing `plane` ----------------
hr = alt.layer(
    cls_rules, ms_band, ms_ref, ref_label,        # CHANGED (5)
    wd_track, cls_letters, hr_labels,
    sun_pt, sun_txt, you,
).properties(width=470, height=440,
             title=alt.Title("Hertzsprung-Russell diagram",
                             color="#f0f0f0"))

# ---- CHANGED (3): hconcat uses `hr` instead of `plane` ---------------
chart = alt.hconcat(portrait, hr).add_params(
    m_sel, a_sel).configure(background="#000000").configure_view(
    fill="#000000", stroke=None)

# ---- CHANGED (4): width="content" replaces deprecated argument -------
# Was:   st.altair_chart(chart, use_container_width=False)
st.altair_chart(chart, width="content")