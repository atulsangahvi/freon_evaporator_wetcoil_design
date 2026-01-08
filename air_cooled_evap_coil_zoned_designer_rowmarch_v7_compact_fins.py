# air_cooled_evap_coil_zoned_designer.py
# Serious(er) DX evaporator row-marching model (Superheat + Evaporation only)
# - Row-by-row marching with partial-row handling at SH↔2φ transition
# - Wet-coil enthalpy method in 2φ rows: driving potential is (h_air - h_sat(Ts))
# - Dry sensible ε–NTU in superheat rows
# - Accurate geometry areas (tube OD, fin 2-face area, fin pitch, exposed tube)
# - Air-side h: Zukauskas crossflow (same family as your condenser code)
# - Refrigerant-side h:
#     * vapor: Dittus–Boelter
#     * 2φ boiling: Shah-style simplified enhancement on liquid D–B (pragmatic)
# - Refrigerant velocity: v = G/rho  (FIXED)
# - Δp: Darcy–Weisbach; 2φ via homogeneous mixture properties
# - Insufficiency: only (Q_achieved < Q_required) and/or (SH_achieved < SH_required)
#
# NOTE: This is a design tool. Eurovent certification ultimately depends on lab testing.
# This model is meant to be physically consistent and useful for engineering iteration.

import math
from math import pi, sqrt, tanh
import io
import traceback

import numpy as np
import pandas as pd
import streamlit as st

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# CoolProp
try:
    from CoolProp.CoolProp import PropsSI
    HAS_CP = True
except Exception:
    HAS_CP = False

# ---------------- Constants ----------------
P_ATM = 101325.0
R_DA  = 287.055
CP_DA = 1006.0
CP_V  = 1860.0
H_LV0 = 2501000.0
INCH  = 0.0254
MM    = 1e-3

def K(tC): return tC + 273.15

# ---------------- Psychrometrics ----------------
def psat_water_Pa(T_C: float) -> float:
    return 611.21 * math.exp((18.678 - T_C/234.5) * (T_C/(257.14 + T_C)))

def W_from_T_RH(T_C: float, RH_pct: float, P: float = P_ATM) -> float:
    RH = max(min(RH_pct, 100.0), 0.1) / 100.0
    Psat = psat_water_Pa(T_C)
    Pv = RH * Psat
    return 0.62198 * Pv / max(P - Pv, 1.0)

def W_from_T_WB(Tdb_C: float, Twb_C: float, P: float = P_ATM) -> float:
    W_sat_wb = W_from_T_RH(Twb_C, 100.0, P)
    h_fg_wb = 2501000.0 - 2369.0 * Twb_C
    numer = (W_sat_wb * (h_fg_wb + CP_V * Twb_C) - CP_DA * (Tdb_C - Twb_C))
    denom = (h_fg_wb + CP_V * Tdb_C)
    W = numer / max(1e-9, denom)
    return max(0.0, W)

def h_moist_J_per_kg_da(T_C: float, W: float) -> float:
    return 1000.0*1.006*T_C + W*(H_LV0 + 1000.0*1.86*T_C)

def cp_moist_J_per_kgK(T_C: float, W: float) -> float:
    return CP_DA + W*CP_V

def rho_moist_kg_m3(T_C: float, W: float, P: float = P_ATM) -> float:
    return P / (R_DA * K(T_C) * (1.0 + 1.6078*W))

def RH_from_T_W(T_C: float, W: float, P: float = P_ATM) -> float:
    Pv = W*P/(0.62198 + W)
    Ps = psat_water_Pa(T_C)
    return max(0.1, min(100.0, 100.0*Pv/max(Ps,1e-9)))

def wb_from_T_W(Tdb_C: float, W_target: float, P: float = P_ATM) -> float:
    lo, hi = -20.0, Tdb_C
    for _ in range(50):
        mid = 0.5*(lo+hi)
        W_mid = W_from_T_WB(Tdb_C, mid, P)
        if W_mid > W_target:
            hi = mid
        else:
            lo = mid
    return 0.5*(lo+hi)

# ---------------- Geometry / Areas ----------------
def geometry_areas(face_W, face_H, Nr, St, Do, tf, FPI, Sl=None):
    """
    Geometry as per user's coil construction (as requested):
      - Tube longitudinal pitch Sl is along face width -> tubes_per_row = floor(face_W / Sl)
      - Number of fins = floor(face_H / fin_pitch) where fin_pitch = (1/FPI)*inch
      - Coil depth = St * Nr
      - Fin "length" = face width; fin "depth" = coil depth; fin area uses width*depth (two faces)
    """
    face_area = face_W*face_H
    fin_pitch = (1.0/FPI)*INCH
    fins = max(int(math.floor(face_H / max(fin_pitch, 1e-9))), 1)

    if Sl is None or Sl <= 0:
        raise ValueError("Longitudinal pitch Sl must be provided (>0).")
    tubes_per_row = max(int(math.floor(face_W / max(Sl, 1e-9))), 1)

    N_tubes = tubes_per_row * Nr
    L_tube = face_W

    depth = St * Nr  # as requested

    # two-face fin area based on fin planform (width x depth) minus tube holes
    A_holes_one_fin = N_tubes * (pi*(Do/2.0)**2)
    A_fin_one = max(2.0*(face_W*depth - A_holes_one_fin), 0.0)
    A_fin_total = A_fin_one * fins

    # bare tube exposure between fins (fins stacked along height)
    exposed_frac = max((fin_pitch - tf)/max(fin_pitch,1e-9), 0.0)
    A_bare = N_tubes * (pi*Do*L_tube) * exposed_frac

    Ao = A_fin_total + A_bare
    Arow = Ao/max(1, Nr)

    # min free-flow area (simple)
    fin_blockage = min(tf/max(fin_pitch,1e-9), 0.95)
    tube_blockage = min(A_holes_one_fin/max(face_area,1e-9), 0.5)
    Amin = max(face_area*(1.0 - fin_blockage - tube_blockage), 1e-4)

    return dict(face_area=face_area, fin_pitch=fin_pitch, fins=fins,
                tubes_per_row=tubes_per_row, N_tubes=N_tubes, L_tube=L_tube,
                depth=depth, A_fin=A_fin_total, A_bare=A_bare,
                Ao=Ao, Arow=Arow, Amin=Amin)

# ---------------- Air-side correlations ----------------
def mu_air_Pas(T_C: float) -> float:
    T = K(T_C)
    return 1.716e-5 * ((T/273.15)**1.5) * ((273.15+110.4)/(T+110.4))

def k_air_W_mK(T_C: float) -> float:
    # simple linear approx
    return 0.024 + (0.027 - 0.024) * (T_C/40.0)

def airside_compact_htc_dp(mdot_air, face_W, face_H, depth, fin_pitch, tf,
                           T_air_C, W_air, P: float = P_ATM,
                           fin_type: str = "Wavy (no louvers)",
                           louver_angle_deg: float = 27.0,
                           louver_cuts_per_row: int = 8,
                           h_mult_wavy: float = 1.15,
                           dp_mult_wavy: float = 1.20):
    """
    Air-side model for stacked plate fins with flow through depth.

    fin_type:
      - "Wavy (no louvers)": parallel-plate duct Nu + Darcy friction with empirical multipliers (waviness/ripple).
      - "Wavy + Louvers": Kim & Bullard style j/f correlation form (as summarized in a louvered-fin review).
        Uses Re based on louver pitch per row. We approximate louver pitch per row as (depth/Nrows_est)/cuts.

    Returns: h [W/m²-K], dp [Pa], meta dict.
    """
    rho = rho_moist_kg_m3(T_air_C, W_air, P)
    mu = mu_air_Pas(T_air_C)
    k  = k_air_W_mK(T_air_C)
    cp = cp_moist_J_per_kgK(T_air_C, W_air)
    Pr = cp*mu/max(k, 1e-12)

    s_f = fin_pitch
    s_gap = max(s_f - tf, 1e-6)
    A_face = face_W*face_H

    sigma = max(min(s_gap/s_f, 0.98), 0.02)
    A_min = max(A_face*sigma, 1e-6)

    G = mdot_air/A_min
    V = G/max(rho, 1e-9)
    Dh = 2.0*s_gap
    Re_Dh = G*Dh/max(mu, 1e-12)

    if fin_type == "Wavy (no louvers)":
        if Re_Dh < 2300:
            Nu = 7.54
            f_D = 96.0/max(Re_Dh, 1e-9)     # Darcy, parallel-plate approx
        else:
            Nu = 0.023*(Re_Dh**0.8)*(Pr**0.4)
            f_D = 0.3164*(Re_Dh**-0.25)

        h = (Nu*k/max(Dh,1e-9))*h_mult_wavy
        dp_core = f_D*(depth/max(Dh,1e-9))*(rho*V*V/2.0)*dp_mult_wavy
        dp_minor = (0.5+1.0)*(rho*V*V/2.0)

        return h, (dp_core+dp_minor), {"model":"duct+wavy","Re":Re_Dh,"Dh":Dh,"A_min":A_min,"sigma":sigma,"V":V}

    # --- louvered (j/f) model
    Nr_est = max(int(round(depth/0.022)), 1)
    Lp_row_m = max((depth/max(Nr_est,1))/max(louver_cuts_per_row,1), 1e-4)  # m
    Lp = Lp_row_m*1000.0
    Fp = s_f*1000.0
    delta = tf*1000.0
    Fh = face_H*1000.0
    Fd = depth*1000.0
    Lh = max(Fp, 0.1)
    Tp = max(Fp, 0.1)
    Lalpha = float(louver_angle_deg)

    Re_Lp = G*Lp_row_m/max(mu, 1e-12)

    j = (Re_Lp**(-0.487)) * ((Lalpha/90.0)**0.257) * ((Fp/Lp)**(-0.13)) * ((Fh/Lp)**(-0.29)) * ((Fd/Lp)**(-0.235)) * ((Lh/Lp)**(0.68)) * ((Tp/Lp)**(-0.279)) * ((delta/Lp)**(-0.05))
    f = (Re_Lp**(-0.781)) * ((Lalpha/90.0)**0.444) * ((Fp/Lp)**(-1.682)) * ((Fh/Lp)**(-1.22)) * ((Fd/Lp)**(0.818)) * ((Lh/Lp)**(1.97))

    h = j*G*cp/max(Pr**(2.0/3.0), 1e-12)

    f_D = 4.0*f
    dp_core = f_D*(depth/max(Dh,1e-9))*(rho*V*V/2.0)
    dp_minor = (0.5+1.0)*(rho*V*V/2.0)

    return h, (dp_core+dp_minor), {"model":"louver_jf","Re":Re_Lp,"Dh":Dh,"A_min":A_min,"sigma":sigma,"V":V,"j":j,"f":f,"Lp_mm":Lp}

def zukauskas_constants(Re):
    Re = max(Re, 1.0)
    if 1e2 <= Re < 1e3:   C, m = 0.9, 0.4
    elif 1e3 <= Re < 2e5: C, m = 0.27, 0.63
    else:                 C, m = (0.27, 0.63) if Re >= 2e5 else (0.9, 0.4)
    return C, m

def row_correction(Nr):
    return 0.70 if Nr<=1 else (0.80 if Nr==2 else (0.88 if Nr==3 else (0.94 if Nr==4 else 1.00)))

def air_htc_zukauskas(rho, mu, k, Pr, Do, Nr, mdot_air, Amin):
    Vmax = mdot_air/(rho*Amin)
    Re = rho*Vmax*Do/max(mu,1e-12)
    C, m = zukauskas_constants(Re)
    Nu = C*(Re**m)*(Pr**0.36) * row_correction(Nr)
    h = Nu*k/Do
    return h, dict(Vmax=Vmax, Re=Re, Nu=Nu)

def fin_efficiency(h, k_fin, t_fin, Lc):
    if Lc<=0 or t_fin<=0 or k_fin<=0: return 1.0
    m = sqrt(2.0*h/(k_fin*t_fin))
    x = max(m*Lc, 1e-9)
    return math.tanh(x)/x

# ---------------- Pressure drop helpers ----------------
def f_churchill(Re, e_over_D):
    Re = max(1e-9, Re)
    if Re < 2300.0:
        return 64.0/max(1.0, Re)
    A = (2.457 * math.log( (7.0 / max(1.0, Re))**0.9 + 0.27*e_over_D ))**16
    B = (37530.0 / max(1.0, Re))**16
    f = 8.0 * ( ((8.0/max(1.0,Re))**12) + 1.0/((A+B)**1.5) )**(1.0/12.0)
    return max(1e-6, f)

def dp_darcy(mdot, rho, mu, D, L, rough=1.5e-6):
    A = pi*D*D/4.0
    G = mdot/max(A,1e-12)              # kg/s/m²
    v = G/max(rho,1e-9)                # m/s  (FIXED)
    Re = rho*v*D/max(mu,1e-12)
    f = f_churchill(Re, rough/max(D,1e-12))
    dp = f*(L/max(D,1e-12))*(0.5*rho*v*v)
    return dp, Re, f, v

def mix_props_homog(x, rho_v, rho_l, mu_v, mu_l):
    rho_m = 1.0 / (x/max(rho_v,1e-12) + (1.0-x)/max(rho_l,1e-12))
    mu_m = x*mu_v + (1.0-x)*mu_l
    return rho_m, mu_m

# ---------------- Refrigerant-side h ----------------
def h_i_dittus_boelter(mdot, rho, mu, k, cp, D):
    A = pi*D*D/4.0
    v = mdot/(rho*A)
    Re = rho*v*D/max(mu,1e-12)
    Pr = cp*mu/max(k,1e-12)
    Re_eff = max(2300.0, Re)
    Nu = 0.023*(Re_eff**0.8)*(Pr**0.4)
    h = Nu*k/max(D,1e-12)
    return h, Re, Pr, v

def h_i_boiling_shah_like(mdot, x, rho_l, mu_l, k_l, cp_l, D, enhancement=1.8):
    # pragmatic: use liquid single-phase h and boost (typical approach in many quick solvers)
    h_l, Re_l, Pr_l, v_l = h_i_dittus_boelter(mdot, rho_l, mu_l, k_l, cp_l, D)
    h_tp = max(800.0, enhancement*h_l)
    return h_tp, Re_l, Pr_l, v_l

# ---------------- Row-by-row marching ----------------

def dewpoint_C_from_T_W(T_C: float, W: float, P: float = P_ATM) -> float:
    """Invert W(T, RH=100%) to get dewpoint for given humidity ratio."""
    # Wsat(T) is monotonic increasing in typical HVAC range.
    lo, hi = -40.0, 60.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        Wsat = W_from_T_RH(mid, 100.0, P)
        if Wsat >= W:
            hi = mid
        else:
            lo = mid
    return 0.5*(lo+hi)

def T_from_h_W(h_J_per_kg_da: float, W: float) -> float:
    """Solve moist air enthalpy equation for T (°C) given h (J/kg_dry_air) and W."""
    # h = 1000*1.006*T + W*(H_LV0 + 1000*1.86*T)
    denom = 1000.0*(1.006 + 1.86*W)
    if denom <= 0:
        return 0.0
    return (h_J_per_kg_da - W*H_LV0)/denom

def bf_from_Q_const_sink(Q: float, C_air: float, driving: float) -> float:
    """
    For a constant-sink element with bypass-factor form:
      Q = C_air*(1-BF)*driving
    => BF = 1 - Q/(C_air*driving)
    """
    if C_air <= 0 or driving <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - Q/max(C_air*driving, 1e-12)))

def UA_from_BF(BF: float, C_air: float) -> float:
    """UA = NTU*C_air; BF = exp(-NTU) => UA = -ln(BF)*C_air."""
    BF = max(min(BF, 0.999999999), 1e-12)
    return (-math.log(BF))*C_air

# ---------------- Row-by-row marching ----------------
def simulate_evaporator(
    face_W, face_H, Nr, St, Sl, Do, tw, tf, FPI,
    fin_k, tube_k, circuits,
    Vdot_m3_s,
    Tdb_in, W_in,
    Tdb_req, W_req,
    fluid, Tsat_C, SH_req_K, mdot_ref_total, x_in,
    wet_enh=1.35, Rfo=0.0, Rfi=0.0
):
    """
    Row marching with partial-row zoning.

    IMPORTANT (corrected): the incoming air first encounters the refrigerant outlet-side
    *superheat* zone, and only afterwards the evaporating (2φ) zone. This is the usual
    HVAC evaporator arrangement when air and refrigerant are counterflowed to ensure
    sufficient superheat.

    Zoning implemented (in this order along air flow):
      1) Superheat (SH): T_ref decreases from Tsat+SH_req -> Tsat (because we march opposite refrigerant flow)
      2) Evaporation (2φ): quality x decreases from 1.0 -> x_in (same reason)

    Each row can be split into partial segments so a zone can finish mid-row and the next zone can begin
    within the same physical row.

    Each row can be split into partial segments so a zone can finish mid-row and the next zone can begin
    within the same physical row.
    """
    if not HAS_CP:
        raise RuntimeError("CoolProp missing. Add CoolProp>=6.6 to requirements.txt.")

    # ---- Air mass flow (dry-air basis) ----
    rho_in = rho_moist_kg_m3(Tdb_in, W_in)
    mdot_air_total = rho_in*Vdot_m3_s
    mdot_da = mdot_air_total/(1.0+W_in)

    h_in = h_moist_J_per_kg_da(Tdb_in, W_in)
    h_req_out = h_moist_J_per_kg_da(Tdb_req, W_req)
    Q_required = mdot_da*(h_in - h_req_out)

    # ---- Geometry ----
    geom = geometry_areas(face_W, face_H, Nr, St, Do, tf, FPI, Sl=Sl)
    Ao = geom["Ao"]; Arow = geom["Arow"]; Amin = geom["Amin"]
    tubes_per_row = geom["tubes_per_row"]
    L_tube = geom["L_tube"]

    # ---- Air properties (for cp) ----
    cp_a = cp_moist_J_per_kgK(Tdb_in, W_in)

    # ---- Air-side h (dry baseline), dp, fin efficiency ----
    h_air_dry, dp_air, air_meta = airside_compact_htc_dp(
        mdot_air_total, face_W, face_H, geom.get('depth', Nr*St), geom['fin_pitch'], tf,
        Tdb_in, W_in, P_ATM,
        fin_type=fin_type,
        louver_angle_deg=louver_angle_deg,
        louver_cuts_per_row=louver_cuts_per_row,
        h_mult_wavy=h_mult_wavy,
        dp_mult_wavy=dp_mult_wavy
    )
    h_air_wet = h_air_dry*wet_enh

    # fin efficiencies
    Lc = max(0.5*(min(St, Sl) - Do), 1e-6)
    eta_f_dry = fin_efficiency(h_air_dry, fin_k, tf, Lc)
    eta_o_dry = 1.0 - (geom["A_fin"]/max(Ao,1e-12))*(1.0-eta_f_dry)
    eta_f_wet = fin_efficiency(h_air_wet, fin_k, tf, Lc)
    eta_o_wet = 1.0 - (geom["A_fin"]/max(Ao,1e-12))*(1.0-eta_f_wet)

    # ---- Tube geometry & Uo helper ----
    Di = max(Do - 2.0*tw, 1e-5)
    Ao_per_m = pi*Do
    Ai_per_m = pi*Di
    Ao_Ai = Ao_per_m/max(Ai_per_m,1e-12)
    R_wall_per_Ao = (math.log(Do/max(Di,1e-12))/(2*pi*tube_k))/max(Ao_per_m,1e-12)

    def Uo(h_i, h_o, eta_o):
        invU = (1.0/max(eta_o*h_o,1e-12)) + Rfo + Ao_Ai*((1.0/max(h_i,1e-12))+Rfi) + R_wall_per_Ao
        return 1.0/max(invU,1e-12)

    # ---- Refrigerant saturation properties ----
    TsK = K(Tsat_C)
    P_sat = PropsSI("P","T",TsK,"Q",0,fluid)
    rho_l = PropsSI("D","T",TsK,"Q",0,fluid)
    rho_v = PropsSI("D","T",TsK,"Q",1,fluid)
    mu_l  = PropsSI("V","T",TsK,"Q",0,fluid)
    mu_v  = PropsSI("V","T",TsK,"Q",1,fluid)
    cp_l  = PropsSI("C","T",TsK,"Q",0,fluid)
    cp_v  = PropsSI("C","T",TsK,"Q",1,fluid)
    k_l   = PropsSI("L","T",TsK,"Q",0,fluid)
    k_v   = PropsSI("L","T",TsK,"Q",1,fluid)
    h_fg  = PropsSI("H","T",TsK,"Q",1,fluid) - PropsSI("H","T",TsK,"Q",0,fluid)

    # ---- Refrigerant flow per circuit & per-row length per circuit ----
    mdot_ref_total = max(mdot_ref_total, 1e-9)
    circuits = max(int(circuits), 1)
    mdot_ref_c = mdot_ref_total/circuits
    L_total_circ = (tubes_per_row*Nr/circuits)*L_tube
    L_row_circ = L_total_circ/max(Nr,1)

    # ---- Marching state ----
    # Air marches forward
    T_air = Tdb_in
    W_air = W_in
    h_air = h_in

    # Refrigerant is marched *opposite* its physical flow direction so that the air-path sees
# SH first, then 2φ (counterflow-style ordering along the air path).
# We start from the refrigerant outlet condition (superheated vapor) and march toward inlet quality.
zone = "SH"
x_target_in = max(0.0, min(0.9999, x_in))
x = 1.0  # at outlet-side (end of 2φ zone) quality is ~1 before superheat
T_ref = Tsat_C + SH_req_K


    # helpers for saturated air at a given surface temp
    def sat_props_at_Ts(Ts_C: float):
        W_s = W_from_T_RH(Ts_C, 100.0)
        h_s = h_moist_J_per_kg_da(Ts_C, W_s)
        return W_s, h_s

    rows_log = []
    Q_total = 0.0

    # ---- Refrigerant pressure drop (rough) ----
    dp_ref_total = 0.0
    dp_ref_SH = 0.0
    dp_ref_2p = 0.0

    # ---- Air-side pressure drop (same as previous one-shot model) ----
    v_face = Vdot_m3_s/max(geom["face_area"],1e-9)
    fin_pitch = geom.get("fin_pitch", geom.get("s"))
    s_fin = max(1e-6, fin_pitch - tf)
    D_h = 2.0*s_fin
    v_core = v_face*(geom["face_area"]/Amin)
    mu_a = mu_air_Pas(Tdb_in)
    Re_ch = rho_in*v_core*D_h/max(mu_a,1e-12)
    q_dyn = 0.5*rho_in*v_core*v_core
    if Re_ch < 2300:
        fD = 64.0/max(Re_ch,1.0)
    else:
        fD = 0.3164/(Re_ch**0.25)
    L_flow = geom.get('depth', Nr*St)
    dp_air = fD*(L_flow/max(D_h,1e-12))*q_dyn + (0.5+1.0)*q_dyn  # inlet+outlet

    # ---- Core row-march with partial-row splitting ----
    for row in range(1, Nr+1):
        UA_row = None  # will compute per segment because h_i changes with zone
        UA_remaining = 1.0  # fraction of row UA available for additional segments
        row_Q = 0.0
        row_Q_2p = 0.0
        row_Q_SH = 0.0
        seg = 0

        # allow multiple zone segments inside same row
        while UA_remaining > 1e-6 and Q_total < Q_required - 1e-9:
            seg += 1

            # --- determine local surface temperature proxy ---
            if zone == "2φ":
                Ts_surf = Tsat_C
            else:
                # SH (or DONE) segment surface proxy
                Ts_surf = T_ref  # simple proxy for SH segment surface

            # --- wet/dry decision based on dew point ---
            Tdp = dewpoint_C_from_T_W(T_air, W_air)
            is_wet = (Ts_surf < Tdp - 1e-6)

            h_o = h_air_wet if is_wet else h_air_dry
            eta_o = eta_o_wet if is_wet else eta_o_dry

            # --- inside HTC and Uo ---
            if zone == "2φ":
                h_i, Re_i, Pr_i, v_i = h_i_boiling_shah_like(mdot_ref_c, x, rho_l, mu_l, k_l, cp_l, Di, enhancement=1.8)
                # mixture props for dp
                rho_m, mu_m = mix_props_homog(max(min(x,0.999),0.001), rho_v, rho_l, mu_v, mu_l)
                v_ref_seg = mdot_ref_c/(rho_m*(pi*Di*Di/4.0))
            else:
                h_i, Re_i, Pr_i, v_i = h_i_dittus_boelter(mdot_ref_c, rho_v, mu_v, k_v, cp_v, Di)
                v_ref_seg = v_i

            U = Uo(h_i, h_o, eta_o)
            UA_full = U*Arow
            UA_seg_avail = UA_full*UA_remaining

            # --- compute full-segment air-side leaving state using BF model ---
            C_air = mdot_da*cp_moist_J_per_kgK(T_air, W_air)

            if is_wet:
                W_s, h_s = sat_props_at_Ts(Ts_surf)
                BF = math.exp(-UA_seg_avail/max(C_air,1e-12))
                h_out_full = h_s + BF*(h_air - h_s)
                W_out_full = W_s + BF*(W_air - W_s)
                T_out_full = T_from_h_W(h_out_full, W_out_full)
                # clamp to saturation
                W_out_full = min(W_out_full, W_from_T_RH(T_out_full, 100.0))
                Q_full = mdot_da*(h_air - h_out_full)
                driving = max(h_air - h_s, 1e-9)
                # for partial UA solve we use enthalpy driving
                def apply_BF(BF_eff):
                    h_out = h_s + BF_eff*(h_air - h_s)
                    W_out = W_s + BF_eff*(W_air - W_s)
                    T_out = T_from_h_W(h_out, W_out)
                    W_out = min(W_out, W_from_T_RH(T_out, 100.0))
                    return T_out, W_out, h_out
            else:
                BF = math.exp(-UA_seg_avail/max(C_air,1e-12))
                T_out_full = Ts_surf + BF*(T_air - Ts_surf)
                W_out_full = W_air
                h_out_full = h_moist_J_per_kg_da(T_out_full, W_out_full)
                Q_full = mdot_da*cp_moist_J_per_kgK(T_air, W_air)*(T_air - T_out_full)
                driving = max(T_air - Ts_surf, 1e-9)
                def apply_BF(BF_eff):
                    T_out = Ts_surf + BF_eff*(T_air - Ts_surf)
                    W_out = W_air
                    h_out = h_moist_J_per_kg_da(T_out, W_out)
                    return T_out, W_out, h_out

            # --- limit by remaining refrigerant-zone duty ---
            if zone == "SH":
                # marching opposite ref flow: remove superheat down to Tsat
                Q_zone_need = mdot_ref_total*cp_v*max(0.0, T_ref - Tsat_C)
            elif zone == "2φ":
                # marching opposite ref flow: reduce quality down to inlet target x_in
                Q_zone_need = mdot_ref_total*h_fg*max(0.0, x - x_target_in)
            else:
                Q_zone_need = 0.0

            Q_rem = max(0.0, Q_required - Q_total)
            Q_take = min(Q_full, Q_zone_need, Q_rem)

            # If refrigerant-side zone is exhausted, stop segmenting to avoid infinite loops
            if Q_take <= 1e-12:
                break

            # --- if we don't take the full Q, compute effective BF and UA fraction used ---
            if Q_take < Q_full - 1e-9:
                BF_eff = bf_from_Q_const_sink(Q_take, C_air, driving)
                UA_eff = UA_from_BF(BF_eff, C_air)
                frac_used = min(1.0, UA_eff/max(UA_seg_avail,1e-12))
                T_air_out, W_air_out, h_air_out = apply_BF(BF_eff)
            else:
                frac_used = 1.0
                T_air_out, W_air_out, h_air_out = T_out_full, W_out_full, h_out_full

            # update air
            T_air, W_air, h_air = T_air_out, W_air_out, h_air_out

            # update totals
            Q_total += Q_take
            row_Q += Q_take
            if zone == "2φ":
                row_Q_2p += Q_take
            else:
                row_Q_SH += Q_take

            # update refrigerant (marched opposite to physical ref flow)
            if zone == "SH":
                T_new = T_ref - Q_take/max(mdot_ref_total*cp_v, 1e-12)
                if T_new <= Tsat_C + 1e-9:
                    # superheat finished within this segment; enter 2φ zone for remaining UA
                    T_ref = Tsat_C
                    zone = "2φ"
                    x = 1.0
                else:
                    T_ref = T_new
            elif zone == "2φ":
                x_new = x - Q_take/max(mdot_ref_total*h_fg, 1e-12)
                if x_new <= x_target_in + 1e-9:
                    # reached inlet condition; no more refrigerant-side capacity beyond this boundary
                    x = x_target_in
                    zone = "DONE"
                else:
                    x = x_new
            else:
                # DONE: no more ref-side duty available
                pass

            # refrigerant dp: allocate by UA fraction used as proxy for length fraction
            L_seg = L_row_circ * (UA_remaining*frac_used)
            if zone == "2φ":
                rho_m, mu_m = mix_props_homog(max(min(x,0.999),0.001), rho_v, rho_l, mu_v, mu_l)
                dp, Re_d, f, v = dp_darcy(mdot_ref_c, rho_m, mu_m, Di, L_seg)
                dp_ref_total += dp; dp_ref_2p += dp
            else:
                dp, Re_d, f, v = dp_darcy(mdot_ref_c, rho_v, mu_v, Di, L_seg)
                dp_ref_total += dp; dp_ref_SH += dp

            # reduce remaining UA fraction
            UA_remaining *= (1.0 - frac_used)

            rows_log.append(dict(
                row=row,
                seg=seg,
                regime=("2φ" if (zone=="2φ" or (zone=="SH" and row_Q_2p>0 and row_Q_SH==0 and x<1.0)) else "SH"),
                zone_at_start=("2φ" if (row_Q_SH==0 and seg==1 and x_in<1.0) else ""),
                wet=is_wet,
                Ts_surf=Ts_surf,
                Tdp_in=Tdp,
                UA_row=UA_full,
                UA_used=UA_seg_avail*frac_used,
                frac_used=frac_used,
                Q_kW=Q_take/1000.0,
                Q_row_2p_kW=row_Q_2p/1000.0,
                Q_row_SH_kW=row_Q_SH/1000.0,
                T_air_out=T_air,
                W_air_out=W_air,
                T_ref=T_ref,
                x=(x if zone=="2φ" else None),
                Re_i=Re_i,
                v_ref=v_ref_seg
            ))

            # stop if SH target achieved
            if zone == "SH" and T_ref >= Tsat_C + SH_req_K - 1e-6:
                break

        # if full duty achieved, stop
        if Q_total >= Q_required - 1e-6:
            break

    # Final achieved air condition
    T_out = T_air
    W_out = W_air
    RH_out = RH_from_T_W(T_out, W_out)
    WB_out = wb_from_T_W(T_out, W_out)

    # Achieved SH
    SH_ach = 0.0
    if zone == "SH":
        SH_ach = max(0.0, T_ref - Tsat_C)
    else:
        SH_ach = 0.0

    insuff = []
    if Q_total + 1e-6 < Q_required:
        insuff.append("Capacity shortfall")
    if SH_ach + 1e-6 < SH_req_K:
        insuff.append("Superheat shortfall")

    summary = {
        "Q_required_kW": Q_required/1000.0,
        "Q_achieved_kW": Q_total/1000.0,
        "Q_sensible_est_kW": (mdot_da*cp_moist_J_per_kgK(Tdb_in, W_in)*(Tdb_in - T_out))/1000.0,
        "Air_out_DB_C": T_out,
        "Air_out_WB_C": WB_out,
        "Air_out_RH_pct": RH_out,
        "Rows_used": (rows_log[-1]["row"] if rows_log else 0),
        "Segments_logged": len(rows_log),
        "Insufficiency": "None" if not insuff else ", ".join(insuff),
        "Tsat_C": Tsat_C,
        "SH_req_K": SH_req_K,
        "SH_ach_K": SH_ach,
        "x_in": x_in,
        "Ao_total_m2": Ao,
        "A_fin_total_m2": geom.get('A_fin'),
        "A_bare_total_m2": geom.get('A_bare'),
        "fins_count": geom.get('fins'),
        "fin_pitch_m": geom.get('fin_pitch'),
        "tubes_per_row": geom.get('tubes_per_row'),
        "total_tubes": geom.get('N_tubes'),
        "tube_length_total_all_tubes_m": geom.get('N_tubes',0)*geom.get('L_tube',0),
        "tube_length_per_row_total_m": geom.get('tubes_per_row',0)*geom.get('L_tube',0),
        "tubes_per_circuit": (geom.get('N_tubes',0)/max(int(circuits),1)),
        "tubes_per_circuit_warning": ("NON-INTEGER: circuit split required" if abs((geom.get('N_tubes',0)/max(int(circuits),1)) - round(geom.get('N_tubes',0)/max(int(circuits),1)))>1e-9 else "OK"),
        "air_model": air_meta.get('model'),
        "air_Re": air_meta.get('Re'),
        "air_Amin_m2": air_meta.get('A_min'),
        "air_Dh_m": air_meta.get('Dh'),
        "air_dp_Pa": dp_air,
        "depth_m": geom.get('depth', Nr*St),
        "Amin_m2": Amin,
        "Arow_m2": Arow,
        "Tube_length_per_row_per_circuit_m": L_row_circ,
        "mdot_ref_total_kg_s": mdot_ref_total,
        "mdot_ref_per_circuit_kg_s": mdot_ref_c,
        "Ref_dp_total_kPa": dp_ref_total/1000.0,
        "Ref_dp_SH_kPa": dp_ref_SH/1000.0,
        "Ref_dp_2p_kPa": dp_ref_2p/1000.0,
        "Air_dp_Pa": dp_air,
        "Air_face_velocity_m_s": v_face,
        "Air_Re_min_channel": Re_ch,
        "h_air_dry_W_m2K": h_air_dry,
        "h_air_wet_W_m2K": h_air_wet,
        "eta_o_dry": eta_o_dry,
        "eta_o_wet": eta_o_wet,
        "P_sat_bar": P_sat/1e5,
        "h_fg_kJ_kg": h_fg/1000.0,
    }

    df = pd.DataFrame(rows_log)
    return df, summary

# ---------------- PDF ----------------
def build_pdf(inputs_dict, rows_df, summary_dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("DX Evaporator — Row Marching ε–NTU Report", styles["Title"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Inputs", styles["Heading2"]))
    inp_items = [[k, str(v)] for k,v in inputs_dict.items()]
    t = Table([["Parameter","Value"]] + inp_items, colWidths=[190, 310])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#f0f0f0")),
        ('GRID',(0,0),(-1,-1),0.25,colors.grey),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),8),
    ]))
    story.append(t); story.append(Spacer(1,10))

    story.append(Paragraph("Row Marching Table", styles["Heading2"]))
    if len(rows_df)==0:
        story.append(Paragraph("No rows computed.", styles["Normal"]))
    else:
        df2 = rows_df.copy()
        # limit columns to readable set
        cols = [c for c in ["row","seg","regime","wet","Q_kW","T_air_out","W_air_out","T_ref","x","frac_used","UA_used","Re_i","v_ref"] if c in df2.columns]
        df2 = df2[cols]
        data = [cols] + df2.round(4).values.tolist()
        tz = Table(data, repeatRows=1)
        tz.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#f0f0f0")),
            ('GRID',(0,0),(-1,-1),0.25,colors.grey),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),7),
        ]))
        story.append(tz); story.append(Spacer(1,10))

    story.append(Paragraph("Summary", styles["Heading2"]))
    sum_items = [[k, str(v)] for k,v in summary_dict.items()]
    ts = Table([["Metric","Value"]] + sum_items, colWidths=[210, 290])
    ts.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#f0f0f0")),
        ('GRID',(0,0),(-1,-1),0.25,colors.grey),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),8),
    ]))
    story.append(ts)

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="DX Evaporator — Row Marching ε–NTU", layout="wide")
st.title("DX Evaporator — Row Marching ε–NTU (Superheat + Evap)")

MAT_K = {"Copper": 380.0, "Aluminum": 205.0, "Steel": 50.0, "CuNi 90/10": 29.0}

st.header("Geometry & Materials")
c1,c2,c3,c4 = st.columns(4)
with c1:
    face_W = st.number_input("Face width (m)", 0.2, 4.0, 1.2, 0.01, format="%.2f")
with c2:
    face_H = st.number_input("Face height (m)", 0.2, 4.0, 0.85, 0.01, format="%.2f")
with c3:
    St_mm = st.number_input("Row-to-row pitch (mm)", 10.0, 60.0, 22.0, 0.01, format="%.2f")
with c4:
    Sl_mm = st.number_input("Longitudinal pitch (mm)", 10.0, 60.0, 24.40, 0.01, format="%.2f")

c5,c6,c7,c8 = st.columns(4)
with c5:
    Nr = st.number_input("Rows", 1, 20, 4, 1)
with c6:
    Do_mm = st.number_input("Tube OD (mm)", 5.0, 20.0, 9.53, 0.01, format="%.2f")
with c7:
    tw_mm = st.number_input("Tube thickness (mm)", 0.2, 2.0, 0.50, 0.01, format="%.2f")
with c8:
    FPI = st.number_input("FPI (1/in)", 4.0, 24.0, 10.0, 0.5)
    fin_type = st.selectbox("Fin type", ["Wavy (no louvers)", "Wavy + Louvers"], index=0)
    if fin_type == "Wavy + Louvers":
        louver_angle_deg = st.number_input("Louver angle (deg)", 0.0, 60.0, 27.0, 0.1)
        louver_cuts_per_row = st.number_input("Louvers per row depth (cuts)", 1, 40, 8, 1)
    else:
        louver_angle_deg = 27.0
        louver_cuts_per_row = 8
    h_mult_wavy = st.number_input("Wavy enhancement multiplier for h", 1.0, 3.0, 1.15, 0.01)
    dp_mult_wavy = st.number_input("Wavy enhancement multiplier for Δp", 1.0, 5.0, 1.20, 0.01)

c9,c10,c11,c12 = st.columns(4)
with c9:
    tf_mm = st.number_input("Fin thickness (mm)", 0.06, 0.30, 0.12, 0.01, format="%.2f")
with c10:
    fin_mat = st.selectbox("Fin material", ["Aluminum","Copper","Steel"])
with c11:
    tube_mat = st.selectbox("Tube material", ["Copper","Aluminum","Steel","CuNi 90/10"])
with c12:
    circuits = st.number_input("Circuits", 2, 64, 8, 1)

fin_k = MAT_K[fin_mat]
tube_k = MAT_K[tube_mat]

St = St_mm*MM
Sl = Sl_mm*MM
Do = Do_mm*MM
tw = tw_mm*MM
tf = tf_mm*MM

st.header("Air Side")
mode_flow = st.radio("Air flow input", ["Face velocity (m/s)", "Volume flow (m³/h)"], horizontal=True)
if mode_flow == "Face velocity (m/s)":
    v_face = st.number_input("Face velocity (m/s)", 0.2, 6.0, 2.5, 0.1, format="%.2f")
    Vdot = v_face*face_W*face_H
else:
    Vdot_h = st.number_input("Air volume flow (m³/h)", 500.0, 50000.0, 7000.0, 10.0, format="%.0f")
    Vdot = Vdot_h/3600.0
    v_face = Vdot/max(face_W*face_H,1e-9)

mode_in = st.radio("Inlet air input", ["DB + RH", "DB + WB"], horizontal=True)
if mode_in == "DB + RH":
    Tdb_in = st.number_input("DB in (°C)", 0.0, 55.0, 24.1, 0.1, format="%.1f")
    RH_in = st.number_input("RH in (%)", 5.0, 100.0, 50.0, 0.5, format="%.1f")
    W_in = W_from_T_RH(Tdb_in, RH_in)
    Twb_in = wb_from_T_W(Tdb_in, W_in)
else:
    Tdb_in = st.number_input("DB in (°C)", 0.0, 55.0, 24.1, 0.1, format="%.1f")
    Twb_in = st.number_input("WB in (°C)", -5.0, 40.0, 17.8, 0.1, format="%.1f")
    W_in = W_from_T_WB(Tdb_in, Twb_in)
    RH_in = RH_from_T_W(Tdb_in, W_in)

mode_req = st.radio("Required outlet air input", ["DB + RH", "DB + WB"], horizontal=True)
if mode_req == "DB + RH":
    Tdb_req = st.number_input("Required DB out (°C)", -10.0, 40.0, 13.5, 0.1, format="%.1f")
    RH_req = st.number_input("Required RH out (%)", 5.0, 100.0, 95.0, 0.5, format="%.1f")
    W_req = W_from_T_RH(Tdb_req, RH_req)
    Twb_req = wb_from_T_W(Tdb_req, W_req)
else:
    Tdb_req = st.number_input("Required DB out (°C)", -10.0, 40.0, 13.5, 0.1, format="%.1f")
    Twb_req = st.number_input("Required WB out (°C)", -10.0, 40.0, 12.8, 0.1, format="%.1f")
    W_req = W_from_T_WB(Tdb_req, Twb_req)
    RH_req = RH_from_T_W(Tdb_req, W_req)

st.header("Refrigerant Side")
fluid = st.selectbox("Refrigerant (CoolProp)", ["R134a","R410A","R407C","R404A","R32","R22"])
Tsat = st.number_input("Evaporating Tsat (°C)", -25.0, 20.0, 5.0, 0.1, format="%.1f")
SH_req = st.number_input("Required superheat (K)", 0.0, 25.0, 6.0, 0.5, format="%.1f")
mdot_ref = st.number_input("Total refrigerant mass flow (kg/s)", 0.001, 2.0, 0.34, 0.001, format="%.3f")
x_in = st.number_input("Inlet quality after TXV (x_in)", 0.0, 0.95, 0.25, 0.01, format="%.2f")

st.header("Factors")
wet_enh = st.number_input("Wet enhancement factor (air-side)", 1.0, 2.5, 1.35, 0.05, format="%.2f")
Rfo = st.number_input("Air-side fouling (m²·K/W)", 0.0, 0.001, 0.0002, 0.00005, format="%.5f")
Rfi = st.number_input("Tube-side fouling (m²·K/W)", 0.0, 0.001, 0.0001, 0.00005, format="%.5f")

run = st.button("Run design")
if run:
    try:
        df_rows, summary = simulate_evaporator(
            face_W, face_H, int(Nr), St, Sl, Do, tw, tf, float(FPI),
            fin_k, tube_k, int(circuits),
            Vdot,
            Tdb_in, W_in,
            Tdb_req, W_req,
            fluid, Tsat, SH_req, mdot_ref, x_in,
            wet_enh=wet_enh, Rfo=Rfo, Rfi=Rfi
        )

        st.subheader("Results")
        cA,cB,cC = st.columns(3)
        with cA:
            st.metric("Required (kW)", f"{summary['Q_required_kW']:.2f}")
            st.metric("Achieved (kW)", f"{summary['Q_achieved_kW']:.2f}")
        with cB:
            st.metric("Air out DB (°C)", f"{summary['Air_out_DB_C']:.2f}")
            st.metric("Air out WB (°C)", f"{summary['Air_out_WB_C']:.2f}")
        with cC:
            st.metric("Air out RH (%)", f"{summary['Air_out_RH_pct']:.1f}")
            st.metric("Insufficiency", summary["Insufficiency"])

        st.subheader("Row-by-row table")
        if len(df_rows) > 0:
            st.dataframe(df_rows, use_container_width=True)
        else:
            st.info("No row results (check inputs).")

        with st.expander("Summary & intermediate values"):
            st.json(summary)

        # PDF download
        inputs_dict = {
            "Face W (m)": face_W, "Face H (m)": face_H, "Rows": int(Nr),
            "St (mm)": St_mm, "Sl (mm)": Sl_mm, "Tube OD (mm)": Do_mm, "Tube t (mm)": tw_mm,
            "Fin t (mm)": tf_mm, "FPI": float(FPI), "Fin k": fin_k, "Tube k": tube_k,
            "Circuits": int(circuits), "Air flow (m3/s)": Vdot,
            "Air in DB (C)": Tdb_in, "Air in RH (%)": RH_in, "Air in WB (C)": Twb_in,
            "Req out DB (C)": Tdb_req, "Req out RH (%)": RH_req, "Req out WB (C)": Twb_req,
            "Refrigerant": fluid, "Tsat (C)": Tsat, "SH_req (K)": SH_req,
            "mdot_ref (kg/s)": mdot_ref, "x_in": x_in,
            "Wet enh": wet_enh, "Rfo": Rfo, "Rfi": Rfi
        }
        pdf = build_pdf(inputs_dict, df_rows, summary)
        st.download_button("📄 Download report (PDF)", data=pdf, file_name="DX_Evaporator_row_marching_report.pdf", mime="application/pdf")

    except Exception as e:
        st.error(str(e))
        st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))