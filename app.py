"""
app.py — Urban Logistics Optimization Dashboard
Mumbai VRP (Vehicle Routing Problem) Solver & Analytics
Run: streamlit run app.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
import streamlit.components.v1 as components
import json, io, textwrap
from copy import deepcopy

from vrp_engine import (
    build_distance_matrix,
    apply_traffic_multipliers,
    nearest_neighbor_solver,
    two_opt_improve,
    solve_with_ortools,
    calculate_route_metrics,
    calculate_signal_timing,
    compute_baseline_metrics,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ROUTE_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#E91E63",
    "#00BCD4", "#8BC34A",
]

ZONES = ["North", "Central", "South", "East"]

VEHICLE_TYPES = {
    "Diesel Van":   dict(cost_per_km=15.0, co2_per_km=0.270),
    "Diesel Truck": dict(cost_per_km=22.0, co2_per_km=0.450),
    "Electric Van": dict(cost_per_km=10.0, co2_per_km=0.055),
    "Petrol Bike":  dict(cost_per_km=6.0,  co2_per_km=0.110),
    "Electric Bike":dict(cost_per_km=4.0,  co2_per_km=0.040),
    "CNG Auto":     dict(cost_per_km=8.0,  co2_per_km=0.120),
}

TIME_PRESETS = {
    "Morning Rush (7–10 AM)": {"North": 2.5, "Central": 2.8, "South": 2.3, "East": 2.1},
    "Midday (10 AM–4 PM)":    {"North": 1.3, "Central": 1.5, "South": 1.4, "East": 1.2},
    "Evening Rush (4–8 PM)":  {"North": 2.4, "Central": 3.0, "South": 2.6, "East": 2.3},
    "Night (8 PM–7 AM)":      {"North": 1.0, "Central": 1.1, "South": 1.0, "East": 1.0},
    "Custom":                  None,
}

# ── Default Mumbai data ──────────────────────────────────────────────────────
DEFAULT_WAREHOUSES = [
    {"name": "Andheri Warehouse",  "lat": 19.1136, "lon": 72.8697, "capacity": 500, "zone": "Central"},
    {"name": "Dadar Hub",          "lat": 19.0178, "lon": 72.8478, "capacity": 400, "zone": "South"},
]

DEFAULT_VEHICLES = [
    {"name": "Van Alpha",    "capacity": 120, "type": "Diesel Van"},
    {"name": "Van Beta",     "capacity": 100, "type": "Diesel Van"},
    {"name": "E-Van Gamma",  "capacity": 90,  "type": "Electric Van"},
    {"name": "Truck Delta",  "capacity": 200, "type": "Diesel Truck"},
]

DEFAULT_DELIVERY = [
    {"name": "Bandra Kurla Complex", "lat": 19.0596, "lon": 72.8656, "demand": 35, "zone": "Central", "priority": "High"},
    {"name": "Worli Sea Face",       "lat": 19.0176, "lon": 72.8183, "demand": 28, "zone": "South",   "priority": "Medium"},
    {"name": "Colaba Market",        "lat": 18.9067, "lon": 72.8147, "demand": 45, "zone": "South",   "priority": "High"},
    {"name": "Kurla West",           "lat": 19.0728, "lon": 72.8826, "demand": 30, "zone": "East",    "priority": "Medium"},
    {"name": "Malad InOrbit",        "lat": 19.1872, "lon": 72.8486, "demand": 50, "zone": "North",   "priority": "Low"},
    {"name": "Borivali Station",     "lat": 19.2307, "lon": 72.8567, "demand": 25, "zone": "North",   "priority": "Medium"},
    {"name": "Powai Lake",           "lat": 19.1197, "lon": 72.9051, "demand": 40, "zone": "East",    "priority": "High"},
    {"name": "Juhu Beach",           "lat": 19.0995, "lon": 72.8265, "demand": 22, "zone": "Central", "priority": "Low"},
    {"name": "Lower Parel Hub",      "lat": 18.9966, "lon": 72.8302, "demand": 38, "zone": "South",   "priority": "High"},
    {"name": "Chembur Colony",       "lat": 19.0622, "lon": 72.9010, "demand": 33, "zone": "East",    "priority": "Medium"},
    {"name": "Ghatkopar East",       "lat": 19.0866, "lon": 72.9086, "demand": 27, "zone": "East",    "priority": "Low"},
    {"name": "Thane West",           "lat": 19.2183, "lon": 72.9781, "demand": 42, "zone": "East",    "priority": "Medium"},
    {"name": "Vashi Navi Mumbai",    "lat": 19.0771, "lon": 73.0074, "demand": 55, "zone": "East",    "priority": "High"},
    {"name": "Goregaon West",        "lat": 19.1663, "lon": 72.8526, "demand": 31, "zone": "North",   "priority": "Medium"},
]


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "warehouses":        deepcopy(DEFAULT_WAREHOUSES),
        "vehicles":          deepcopy(DEFAULT_VEHICLES),
        "delivery_points":   deepcopy(DEFAULT_DELIVERY),
        "traffic_factors":   {"North": 1.5, "Central": 2.0, "South": 1.8, "East": 1.3},
        "routes":            None,
        "metrics":           None,
        "all_locations":     None,
        "dist_matrix":       None,
        "time_matrix":       None,
        "demands":           None,
        "solver_used":       None,
        "baseline":          None,
        "heuristic_metrics": None,
        "ortools_metrics":   None,
        "depot_idx":         0,
        "signal_route_idx":  0,
        "cycle_time":        90,
        "map_tile":          "cartodbpositron",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mumbai Logistics Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root { --accent: #FF6B35; --card-bg: #1a1d2e; --border: #2d3561; }
.hero-banner {
    background: linear-gradient(135deg,#0f0c29,#302b63,#24243e);
    padding: 28px 32px; border-radius: 14px; margin-bottom: 22px;
}
.hero-banner h1 { margin:0; font-size:2rem; color:#fff; }
.hero-banner p  { margin:6px 0 0; color:#aaa; font-size:.95rem; }
.kpi-box {
    background: var(--card-bg); border:1px solid var(--border);
    border-radius:10px; padding:16px; text-align:center;
}
.kpi-box .val  { font-size:1.7rem; font-weight:700; color:var(--accent); }
.kpi-box .lbl  { font-size:.78rem; color:#999; margin-top:2px; }
.route-chip {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:.75rem; font-weight:600; margin:2px; color:#fff;
}
.section-hdr {
    font-size:.85rem; font-weight:700; color:#888; letter-spacing:.08em;
    text-transform:uppercase; margin:18px 0 8px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

init_state()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🚚 Logistics Optimizer")
        st.markdown("**Mumbai VRP Dashboard**")
        st.divider()

        # ── Warehouses ────────────────────────────────────────────────────
        with st.expander("🏭 Warehouses", expanded=False):
            warehouses = st.session_state.warehouses
            for i, w in enumerate(warehouses):
                st.markdown(f"**{w['name']}**")
                c1, c2 = st.columns(2)
                warehouses[i]["name"]     = c1.text_input("Name",     w["name"],     key=f"wn{i}")
                warehouses[i]["capacity"] = c2.number_input("Cap",    w["capacity"],  key=f"wc{i}", min_value=1)
                c3, c4 = st.columns(2)
                warehouses[i]["lat"]  = c3.number_input("Lat", w["lat"],  key=f"wla{i}", format="%.4f", step=0.001)
                warehouses[i]["lon"]  = c4.number_input("Lon", w["lon"],  key=f"wlo{i}", format="%.4f", step=0.001)
                warehouses[i]["zone"] = st.selectbox("Zone", ZONES, ZONES.index(w["zone"]), key=f"wz{i}")
                if len(warehouses) > 1 and st.button("🗑 Remove", key=f"wrm{i}"):
                    warehouses.pop(i); st.rerun()
                st.divider()

            st.markdown('<div class="section-hdr">Add Warehouse</div>', unsafe_allow_html=True)
            nw_name = st.text_input("Name", "New Warehouse", key="nw_name")
            c1, c2 = st.columns(2)
            nw_lat  = c1.number_input("Lat", 19.10, key="nw_lat", format="%.4f", step=0.001)
            nw_lon  = c2.number_input("Lon", 72.88, key="nw_lon", format="%.4f", step=0.001)
            c3, c4 = st.columns(2)
            nw_cap  = c3.number_input("Cap", 300, key="nw_cap", min_value=1)
            nw_zone = c4.selectbox("Zone", ZONES, key="nw_zone")
            if st.button("➕ Add Warehouse"):
                warehouses.append({"name": nw_name, "lat": nw_lat, "lon": nw_lon,
                                   "capacity": nw_cap, "zone": nw_zone})
                st.rerun()

            # Depot selection
            st.markdown('<div class="section-hdr">Active Depot</div>', unsafe_allow_html=True)
            depot_names = [w["name"] for w in warehouses]
            st.session_state.depot_idx = st.selectbox(
                "Route from", depot_names, key="depot_sel",
                index=min(st.session_state.depot_idx, len(depot_names)-1)
            )
            st.session_state.depot_idx = depot_names.index(st.session_state.depot_idx)

        # ── Vehicles ──────────────────────────────────────────────────────
        with st.expander("🚗 Fleet", expanded=False):
            vehicles = st.session_state.vehicles
            for i, v in enumerate(vehicles):
                st.markdown(f"**{v['name']}** — {v['type']}")
                c1, c2 = st.columns(2)
                vehicles[i]["name"] = c1.text_input("Name", v["name"], key=f"vn{i}")
                vtype = c2.selectbox("Type", list(VEHICLE_TYPES.keys()),
                                     list(VEHICLE_TYPES.keys()).index(v["type"]) if v["type"] in VEHICLE_TYPES else 0,
                                     key=f"vt{i}")
                vehicles[i]["type"]       = vtype
                vehicles[i]["cost_per_km"]= VEHICLE_TYPES[vtype]["cost_per_km"]
                vehicles[i]["co2_per_km"] = VEHICLE_TYPES[vtype]["co2_per_km"]
                vehicles[i]["capacity"]   = st.number_input("Capacity (units)", v["capacity"],
                                                            key=f"vc{i}", min_value=1)
                if len(vehicles) > 1 and st.button("🗑 Remove", key=f"vrm{i}"):
                    vehicles.pop(i); st.rerun()
                st.divider()

            st.markdown('<div class="section-hdr">Add Vehicle</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            nv_name = c1.text_input("Name", "New Vehicle", key="nv_name")
            nv_type = c2.selectbox("Type", list(VEHICLE_TYPES.keys()), key="nv_type")
            nv_cap  = st.number_input("Capacity", 100, key="nv_cap", min_value=1)
            if st.button("➕ Add Vehicle"):
                vt = VEHICLE_TYPES[nv_type]
                vehicles.append({"name": nv_name, "type": nv_type, "capacity": nv_cap,
                                  "cost_per_km": vt["cost_per_km"], "co2_per_km": vt["co2_per_km"]})
                st.rerun()

        # ── Delivery Points ───────────────────────────────────────────────
        with st.expander("📦 Delivery Points", expanded=False):
            dps = st.session_state.delivery_points
            for i, dp in enumerate(dps):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"**{dp['name']}**")
                if c2.button("🗑", key=f"dprm{i}") and len(dps) > 1:
                    dps.pop(i); st.rerun()
                c1, c2, c3 = st.columns(3)
                dps[i]["demand"]   = c1.number_input("Demand", dp["demand"],  key=f"dpd{i}", min_value=1)
                dps[i]["zone"]     = c2.selectbox("Zone", ZONES, ZONES.index(dp["zone"]) if dp["zone"] in ZONES else 0, key=f"dpz{i}")
                dps[i]["priority"] = c3.selectbox("Priority", ["High","Medium","Low"],
                                                  ["High","Medium","Low"].index(dp.get("priority","Medium")), key=f"dpp{i}")

            st.markdown('<div class="section-hdr">Add Delivery Point</div>', unsafe_allow_html=True)
            nd_name = st.text_input("Location Name", "New Point", key="nd_name")
            c1, c2 = st.columns(2)
            nd_lat  = c1.number_input("Lat", 19.08, key="nd_lat", format="%.4f", step=0.001)
            nd_lon  = c2.number_input("Lon", 72.88, key="nd_lon", format="%.4f", step=0.001)
            c3, c4, c5 = st.columns(3)
            nd_dem  = c3.number_input("Demand", 30, key="nd_dem", min_value=1)
            nd_zone = c4.selectbox("Zone", ZONES, key="nd_zone")
            nd_pri  = c5.selectbox("Priority", ["High","Medium","Low"], key="nd_pri")
            if st.button("➕ Add Point"):
                dps.append({"name": nd_name, "lat": nd_lat, "lon": nd_lon,
                             "demand": nd_dem, "zone": nd_zone, "priority": nd_pri})
                st.rerun()

        # ── Traffic ───────────────────────────────────────────────────────
        with st.expander("🚦 Traffic", expanded=True):
            preset = st.selectbox("Time Preset", list(TIME_PRESETS.keys()), key="time_preset")
            if preset != "Custom" and TIME_PRESETS[preset]:
                st.session_state.traffic_factors = deepcopy(TIME_PRESETS[preset])

            tf = st.session_state.traffic_factors
            for zone in ZONES:
                tf[zone] = st.slider(
                    f"{zone} Zone ×", 1.0, 4.0, float(tf.get(zone, 1.5)),
                    step=0.1, key=f"tf_{zone}"
                )

            st.session_state.cycle_time = st.number_input(
                "Signal Cycle Time (sec)", 30, 180, st.session_state.cycle_time, 10
            )

        # ── Solver ────────────────────────────────────────────────────────
        with st.expander("⚙️ Solver", expanded=True):
            solver_choice = st.radio("Algorithm",
                                     ["Heuristic (Fast)", "OR-Tools (Optimal)", "Both"],
                                     key="solver_choice")
            time_limit = st.slider("OR-Tools Time Limit (sec)", 5, 120, 30, 5, key="time_limit")
            apply_2opt = st.checkbox("Apply 2-opt Improvement", True, key="apply_2opt")
            st.session_state.map_tile = st.selectbox(
                "Map Style",
                ["cartodbpositron", "cartodbdark_matter", "openstreetmap"],
                key="map_tile_sel",
            )

            st.divider()
            if st.button("🚀 Solve VRP", type="primary", use_container_width=True):
                solve_vrp(solver_choice, time_limit, apply_2opt)

            if st.button("🔄 Reset to Defaults", use_container_width=True):
                for k in ["warehouses","vehicles","delivery_points","routes","metrics",
                          "all_locations","dist_matrix","time_matrix","demands",
                          "solver_used","baseline","heuristic_metrics","ortools_metrics"]:
                    st.session_state.pop(k, None)
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SOLVE
# ─────────────────────────────────────────────────────────────────────────────
def solve_vrp(solver_choice: str, time_limit: int, apply_2opt: bool):
    warehouses     = st.session_state.warehouses
    vehicles       = st.session_state.vehicles
    delivery_pts   = st.session_state.delivery_points
    traffic_factors= st.session_state.traffic_factors
    depot_idx_raw  = st.session_state.depot_idx

    if not warehouses or not vehicles or not delivery_pts:
        st.error("Need ≥1 warehouse, vehicle, and delivery point."); return

    # Build unified location list: warehouses first, then deliveries
    all_locs = []
    for w in warehouses:
        all_locs.append({**w, "type": "warehouse", "demand": 0})
    for dp in delivery_pts:
        all_locs.append({**dp, "type": "delivery"})

    depot_idx = depot_idx_raw  # index into all_locs (warehouse index)
    demands   = [float(loc.get("demand", 0)) for loc in all_locs]
    zones     = [loc.get("zone", "Central") for loc in all_locs]

    max_cap   = max(v["capacity"] for v in vehicles)
    n_veh     = len(vehicles)

    total_demand = sum(demands)
    if total_demand > max_cap * n_veh:
        st.warning(
            f"⚠️ Total demand ({total_demand:.0f}) exceeds fleet capacity "
            f"({max_cap*n_veh:.0f}). Some deliveries may be unserved."
        )

    with st.spinner("Building distance matrix…"):
        dist_mat = build_distance_matrix(all_locs)
        time_mat = apply_traffic_multipliers(dist_mat, zones, traffic_factors)

    routes_h = routes_o = None
    status_msg = []

    # Heuristic
    if "Heuristic" in solver_choice or "Both" in solver_choice:
        with st.spinner("Running Nearest-Neighbour + 2-opt…"):
            routes_h = nearest_neighbor_solver(dist_mat, demands, max_cap, n_veh, depot_idx)
            if apply_2opt:
                routes_h = two_opt_improve(routes_h, dist_mat)
            m_h = calculate_route_metrics(routes_h, all_locs, dist_mat, time_mat, demands, vehicles)
            st.session_state.heuristic_metrics = m_h
            status_msg.append(f"Heuristic: {len(routes_h)} routes")

    # OR-Tools
    if "OR-Tools" in solver_choice or "Both" in solver_choice:
        with st.spinner("Running OR-Tools CVRP…"):
            routes_o, ort_status = solve_with_ortools(dist_mat, demands, max_cap, n_veh, depot_idx, time_limit)
            if routes_o:
                m_o = calculate_route_metrics(routes_o, all_locs, dist_mat, time_mat, demands, vehicles)
                st.session_state.ortools_metrics = m_o
                status_msg.append(f"OR-Tools: {len(routes_o)} routes")
            else:
                status_msg.append(f"OR-Tools: {ort_status}")

    # Pick best routes to display
    best_routes  = routes_o if routes_o else routes_h
    solver_label = ("OR-Tools" if routes_o and "OR-Tools" in solver_choice else "Heuristic (NN + 2-opt)")
    if "Both" in solver_choice:
        if routes_o and routes_h:
            dist_o = sum(sum(dist_mat[r[k], r[k+1]] for k in range(len(r)-1)) for r in routes_o)
            dist_h = sum(sum(dist_mat[r[k], r[k+1]] for k in range(len(r)-1)) for r in routes_h)
            best_routes  = routes_o if dist_o <= dist_h else routes_h
            solver_label = f"OR-Tools ({dist_o:.1f} km)" if dist_o <= dist_h else f"Heuristic ({dist_h:.1f} km)"

    if not best_routes:
        st.error("No feasible routes found. Try increasing vehicle capacity or count."); return

    metrics  = calculate_route_metrics(best_routes, all_locs, dist_mat, time_mat, demands, vehicles)
    baseline = compute_baseline_metrics(all_locs, dist_mat, time_mat, demands, depot_idx)

    st.session_state.update(dict(
        routes        = best_routes,
        metrics       = metrics,
        all_locations = all_locs,
        dist_matrix   = dist_mat,
        time_matrix   = time_mat,
        demands       = demands,
        solver_used   = solver_label,
        baseline      = baseline,
    ))
    st.success(f"✅ {' | '.join(status_msg)}  →  Displaying: **{solver_label}**")


# ─────────────────────────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────────────────────────
def build_folium_map(metrics, all_locs, tile="cartodbpositron"):
    lats = [l["lat"] for l in all_locs]
    lons = [l["lon"] for l in all_locs]
    m = folium.Map(
        location=[np.mean(lats), np.mean(lons)],
        zoom_start=12,
        tiles=tile,
    )

    # Warehouse markers
    for loc in all_locs:
        if loc["type"] == "warehouse":
            folium.Marker(
                [loc["lat"], loc["lon"]],
                popup=folium.Popup(
                    f"<b>🏭 {loc['name']}</b><br>Capacity: {loc['capacity']} units", max_width=200
                ),
                icon=folium.Icon(color="red", icon="industry", prefix="fa"),
                tooltip=f"🏭 {loc['name']}",
            ).add_to(m)

    # Route layers
    for i, metric in enumerate(metrics):
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        grp   = folium.FeatureGroup(name=f"Route {metric['route_id']}: {metric['vehicle_name']}")
        coords= [[c["lat"], c["lon"]] for c in metric["coords"]]

        folium.PolyLine(
            coords, color=color, weight=5, opacity=0.85,
            tooltip=f"Route {metric['route_id']} | {metric['distance_km']} km | ₹{metric['cost_inr']}",
        ).add_to(grp)

        # Direction arrows (every other segment)
        for k in range(0, len(coords)-1, 2):
            mid = [(coords[k][0]+coords[k+1][0])/2, (coords[k][1]+coords[k+1][1])/2]
            folium.RegularPolygonMarker(
                mid, color=color, fill=True, fill_color=color,
                fill_opacity=0.9, number_of_sides=3, radius=5,
            ).add_to(grp)

        # Stop markers with sequence numbers
        for j, (stop, coord) in enumerate(
            zip(metric["stops"][1:-1], metric["coords"][1:-1]), 1
        ):
            # Circle + number
            folium.CircleMarker(
                [coord["lat"], coord["lon"]], radius=13,
                color=color, weight=3, fill=True, fill_color=color, fill_opacity=0.25,
                popup=folium.Popup(
                    f"<b>{j}. {stop}</b><br>Route {metric['route_id']}<br>"
                    f"Demand: {metric['route_demands'][j]} units", max_width=180
                ),
                tooltip=f"{j}. {stop}",
            ).add_to(grp)
            folium.Marker(
                [coord["lat"], coord["lon"]],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font-size:10px;color:white;background:{color};'
                        f'border-radius:50%;width:18px;height:18px;text-align:center;'
                        f'line-height:18px;font-weight:700;">{j}</div>'
                    ),
                    icon_size=(18, 18), icon_anchor=(9, 9),
                ),
            ).add_to(grp)
        grp.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#ccc", margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

def _colors_for(n): return [ROUTE_COLORS[i % len(ROUTE_COLORS)] for i in range(n)]


def chart_cost_distance(metrics):
    names  = [f"R{m['route_id']} {m['vehicle_name']}" for m in metrics]
    costs  = [m["cost_inr"]      for m in metrics]
    dists  = [m["distance_km"]   for m in metrics]
    colors = _colors_for(len(metrics))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Cost (₹)", "Distance (km)"))
    fig.add_trace(go.Bar(x=names, y=costs,  marker_color=colors, name="Cost"),  row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=dists,  marker_color=colors, name="Dist"),  row=1, col=2)
    fig.update_layout(**DARK_LAYOUT, title="Route Cost & Distance", showlegend=False, height=340)
    return fig


def chart_utilization(metrics):
    names = [f"R{m['route_id']}" for m in metrics]
    utils = [m["utilization_pct"] for m in metrics]
    colors= [("#2ECC71" if u >= 70 else "#F39C12" if u >= 40 else "#E74C3C") for u in utils]
    fig = go.Figure(go.Bar(x=names, y=utils, marker_color=colors,
                           text=[f"{u:.0f}%" for u in utils], textposition="outside"))
    fig.update_layout(**DARK_LAYOUT, title="Vehicle Utilization (%)", yaxis_range=[0,115], height=300)
    fig.add_hline(y=70, line_dash="dot", line_color="white", annotation_text="70% target")
    return fig


def chart_time_analysis(metrics):
    names = [f"R{m['route_id']} {m['vehicle_name']}" for m in metrics]
    times = [m["travel_time_hr"] for m in metrics]
    dists = [m["distance_km"]    for m in metrics]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dists, y=times, mode="markers+text",
        text=names, textposition="top center",
        marker=dict(size=14, color=_colors_for(len(metrics))),
    ))
    fig.update_layout(**DARK_LAYOUT, title="Travel Time vs Distance",
                      xaxis_title="Distance (km)", yaxis_title="Travel Time (hr)", height=320)
    return fig


def chart_co2_breakdown(metrics):
    names    = [f"R{m['route_id']}" for m in metrics]
    base_co2 = [round(m["distance_km"] * m["co2_per_km"], 3)               for m in metrics]
    load_add = [round(m["co2_kg"] / m["load_factor"] * (m["load_factor"]-1), 3) for m in metrics]
    traff_add= [round(m["co2_kg"] - m["distance_km"]*m["co2_per_km"]*m["load_factor"], 3)
                for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Base Emissions",      x=names, y=base_co2, marker_color="#2ECC71"))
    fig.add_trace(go.Bar(name="Load Penalty",        x=names, y=load_add, marker_color="#F39C12"))
    fig.add_trace(go.Bar(name="Stop-and-Go Penalty", x=names, y=[max(0,t) for t in traff_add], marker_color="#E74C3C"))
    fig.update_layout(**DARK_LAYOUT, barmode="stack",
                      title="CO₂ Emissions Breakdown (kg)", height=340)
    return fig


def chart_co2_per_delivery(metrics):
    names = [f"R{m['route_id']}" for m in metrics]
    cpd   = [round(m["co2_kg"] / max(m["num_deliveries"],1), 3) for m in metrics]
    fig   = go.Figure(go.Bar(x=names, y=cpd, marker_color=_colors_for(len(metrics)),
                             text=[f"{v:.3f}" for v in cpd], textposition="outside"))
    fig.update_layout(**DARK_LAYOUT, title="CO₂ per Delivery (kg)", height=300)
    return fig


def chart_distance_heatmap(dist_matrix, all_locs):
    names = [l["name"].split()[0] for l in all_locs]
    fig = go.Figure(go.Heatmap(
        z=dist_matrix, x=names, y=names,
        colorscale="Viridis", text=np.round(dist_matrix, 1).tolist(),
        texttemplate="%{text}", colorbar_title="km",
    ))
    fig.update_layout(**DARK_LAYOUT, title="Distance Matrix Heatmap (km)",
                      height=450, xaxis_nticks=len(names), yaxis_nticks=len(names))
    return fig


def chart_pareto(metrics):
    fig = go.Figure()
    for i, m in enumerate(metrics):
        fig.add_trace(go.Scatter(
            x=[m["co2_kg"]], y=[m["cost_inr"]],
            mode="markers+text",
            text=[f"R{m['route_id']}"],
            textposition="top right",
            marker=dict(size=18, color=ROUTE_COLORS[i % len(ROUTE_COLORS)],
                        symbol="circle", line=dict(width=2, color="white")),
            name=f"R{m['route_id']} {m['vehicle_name']}",
        ))
    fig.update_layout(**DARK_LAYOUT, title="Cost vs CO₂ — Pareto Front",
                      xaxis_title="CO₂ (kg)", yaxis_title="Cost (₹)", height=360)
    return fig


def chart_demand_pie(metrics, all_locs):
    zones  = [l.get("zone","?") for l in all_locs if l["type"]=="delivery"]
    dems   = [l.get("demand",0) for l in all_locs if l["type"]=="delivery"]
    df     = pd.DataFrame({"zone": zones, "demand": dems}).groupby("zone").sum().reset_index()
    fig    = go.Figure(go.Pie(labels=df["zone"], values=df["demand"],
                               hole=0.45, marker_colors=_colors_for(len(df))))
    fig.update_layout(**DARK_LAYOUT, title="Demand Distribution by Zone", height=320)
    return fig


def chart_traffic_impact(metrics, traffic_factors):
    zones  = list(traffic_factors.keys())
    values = [traffic_factors[z] for z in zones]
    colors = [("#2ECC71" if v < 1.5 else "#F39C12" if v < 2.2 else "#E74C3C") for v in values]
    fig    = go.Figure(go.Bar(x=zones, y=values, marker_color=colors,
                               text=[f"×{v:.1f}" for v in values], textposition="outside"))
    fig.update_layout(**DARK_LAYOUT, title="Traffic Multiplier by Zone",
                      yaxis_range=[0, max(values)*1.2+0.5], height=300)
    fig.add_hline(y=1.0, line_dash="dot", line_color="white", annotation_text="Free flow")
    return fig


def chart_vehicle_comparison():
    vt    = VEHICLE_TYPES
    names = list(vt.keys())
    costs = [vt[n]["cost_per_km"] for n in names]
    co2s  = [vt[n]["co2_per_km"]  for n in names]
    fig   = make_subplots(rows=1, cols=2, subplot_titles=("Cost/km (₹)", "CO₂/km (kg)"))
    fig.add_trace(go.Bar(x=names, y=costs, marker_color="#3498DB", name="Cost"), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=co2s,  marker_color="#2ECC71", name="CO₂"),  row=1, col=2)
    fig.update_layout(**DARK_LAYOUT, title="Vehicle Type Comparison", showlegend=False, height=320)
    fig.update_xaxes(tickangle=25)
    return fig


def chart_signal_gantt(signal_records):
    if not signal_records:
        return None
    rows = []
    for s in signal_records:
        rows.append(dict(
            Segment=s["segment"][:30],
            Start=s["arrival_time_min"]-s["travel_time_min"],
            End=s["arrival_time_min"],
            Phase="🟢 Green" if s["green_wave_ok"] else "🔴 Red",
            Score=s["alignment_score"],
        ))
    df = pd.DataFrame(rows)
    colors = {"🟢 Green": "#2ECC71", "🔴 Red": "#E74C3C"}
    fig = go.Figure()
    for _, row in df.iterrows():
        c = colors[row["Phase"]]
        fig.add_trace(go.Bar(
            x=[row["End"]-row["Start"]], base=[row["Start"]],
            y=[row["Segment"]], orientation="h",
            marker_color=c, name=row["Phase"],
            text=f"{row['Score']:.0f}%", textposition="inside",
            showlegend=False,
        ))
    fig.update_layout(**DARK_LAYOUT, title="Signal Phase Timeline (min)",
                      xaxis_title="Minutes from Route Start", barmode="stack",
                      height=max(300, 40*len(signal_records)))
    return fig


def chart_alignment_scores(signal_records):
    if not signal_records:
        return None
    segs   = [s["segment"][:28] for s in signal_records]
    scores = [s["alignment_score"] for s in signal_records]
    colors = [("#2ECC71" if sc >= 70 else "#F39C12" if sc >= 40 else "#E74C3C") for sc in scores]
    fig    = go.Figure(go.Bar(x=segs, y=scores, marker_color=colors,
                               text=[f"{sc:.0f}" for sc in scores], textposition="outside"))
    fig.update_layout(**DARK_LAYOUT, title="Green-Wave Alignment Score per Segment",
                      yaxis_range=[0,115], xaxis_tickangle=30, height=320)
    fig.add_hline(y=70, line_dash="dot", line_color="white")
    return fig


def chart_comparison(baseline, metrics_h, metrics_o=None):
    labels, costs, co2s, dists = ["Baseline (Direct)"], \
        [baseline["baseline_cost_inr"]], [baseline["baseline_co2_kg"]], \
        [baseline["total_distance_km"]]

    if metrics_h:
        labels.append("Heuristic")
        costs.append(sum(m["cost_inr"]    for m in metrics_h))
        co2s.append( sum(m["co2_kg"]      for m in metrics_h))
        dists.append(sum(m["distance_km"] for m in metrics_h))

    if metrics_o:
        labels.append("OR-Tools")
        costs.append(sum(m["cost_inr"]    for m in metrics_o))
        co2s.append( sum(m["co2_kg"]      for m in metrics_o))
        dists.append(sum(m["distance_km"] for m in metrics_o))

    palette = ["#888888", "#3498DB", "#2ECC71"][:len(labels)]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Total Cost (₹)","Total CO₂ (kg)","Total Distance (km)"))
    fig.add_trace(go.Bar(x=labels, y=costs, marker_color=palette, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=co2s,  marker_color=palette, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=dists, marker_color=palette, showlegend=False), row=1, col=3)
    fig.update_layout(**DARK_LAYOUT, title="Solver Comparison vs Baseline", height=360)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
def render_kpi_row(metrics, baseline):
    total_cost  = sum(m["cost_inr"]      for m in metrics)
    total_co2   = sum(m["co2_kg"]        for m in metrics)
    total_dist  = sum(m["distance_km"]   for m in metrics)
    total_dels  = sum(m["num_deliveries"] for m in metrics)
    avg_util    = np.mean([m["utilization_pct"] for m in metrics])
    cost_saving = baseline["baseline_cost_inr"] - total_cost
    co2_saving  = baseline["baseline_co2_kg"]   - total_co2

    cols = st.columns(6)
    kpis = [
        ("₹ Total Cost",      f"₹{total_cost:,.0f}",       ""),
        ("🌿 CO₂ Saved",       f"{co2_saving:+.1f} kg",     "vs baseline"),
        ("📏 Total Distance",  f"{total_dist:.1f} km",      ""),
        ("📦 Deliveries",      str(total_dels),             "across all routes"),
        ("🚗 Avg Utilization", f"{avg_util:.0f}%",          "fleet load"),
        ("💰 Cost Saved",      f"₹{cost_saving:,.0f}",      "vs baseline"),
    ]
    for col, (lbl, val, sub) in zip(cols, kpis):
        col.markdown(
            f'<div class="kpi-box">'
            f'<div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div>'
            f'<div style="font-size:.7rem;color:#666">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
def tab_route_map(metrics, all_locs):
    st.subheader("🗺️ Interactive Route Map")
    if not metrics:
        st.info("Solve the VRP first to see routes on the map.")
        return

    tile = st.session_state.map_tile
    m = build_folium_map(metrics, all_locs, tile)
    html_map = m._repr_html_()
    components.html(html_map, height=620, scrolling=False)

    # Route legend table
    st.markdown("##### Route Summary")
    rows = []
    for i, met in enumerate(metrics):
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        badge = (
            f'<span class="route-chip" style="background:{color};">'
            f'R{met["route_id"]}</span>'
        )
        rows.append({
            "Route": badge,
            "Vehicle": met["vehicle_name"],
            "Type": met["vehicle_type"],
            "Stops": " → ".join(met["stops"][1:-1]),
            "Dist (km)": met["distance_km"],
            "Time (hr)": met["travel_time_hr"],
            "Cost (₹)": f'₹{met["cost_inr"]:,.0f}',
            "CO₂ (kg)": met["co2_kg"],
            "Util %": met["utilization_pct"],
        })
    st.markdown(
        pd.DataFrame(rows).to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )


def tab_route_analysis(metrics):
    st.subheader("📊 Route Analysis")
    if not metrics:
        st.info("Run the solver to see analysis.")
        return

    c1, c2 = st.columns(2)
    c1.plotly_chart(chart_cost_distance(metrics), use_container_width=True)
    c2.plotly_chart(chart_utilization(metrics),   use_container_width=True)

    c3, c4 = st.columns(2)
    c3.plotly_chart(chart_time_analysis(metrics),  use_container_width=True)
    c4.plotly_chart(chart_vehicle_comparison(),    use_container_width=True)

    # Per-route detail expanders
    st.markdown("##### Detailed Route Plans")
    for i, met in enumerate(metrics):
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        with st.expander(
            f"Route {met['route_id']} — {met['vehicle_name']} "
            f"| {met['distance_km']} km | {met['num_deliveries']} stops"
        ):
            st.markdown(
                f"**Vehicle:** {met['vehicle_name']} ({met['vehicle_type']})  |  "
                f"**Load:** {met['load']} / {met['capacity']} units ({met['utilization_pct']}%)  |  "
                f"**Cost:** ₹{met['cost_inr']:,.0f}  |  **CO₂:** {met['co2_kg']} kg"
            )
            stop_rows = []
            cum_dist = 0.0
            for j, stop in enumerate(met["stops"]):
                t = "🏭 Depot" if j in (0, len(met["stops"])-1) else "📦 Delivery"
                stop_rows.append({"#": j, "Type": t, "Location": stop,
                                   "Demand": met["route_demands"][j]})
            st.dataframe(pd.DataFrame(stop_rows), use_container_width=True, hide_index=True)


def tab_emissions(metrics):
    st.subheader("🌿 Emissions Dashboard")
    if not metrics:
        st.info("Run the solver to see emissions data.")
        return

    c1, c2 = st.columns(2)
    c1.plotly_chart(chart_co2_breakdown(metrics),    use_container_width=True)
    c2.plotly_chart(chart_co2_per_delivery(metrics), use_container_width=True)

    # Cumulative CO₂ donut
    total_co2 = sum(m["co2_kg"] for m in metrics)
    labels = [f"R{m['route_id']} {m['vehicle_name']}" for m in metrics]
    fig = go.Figure(go.Pie(
        labels=labels, values=[m["co2_kg"] for m in metrics],
        hole=0.5, marker_colors=_colors_for(len(metrics)),
    ))
    fig.add_annotation(text=f"{total_co2:.2f} kg<br><span style='font-size:10px'>Total CO₂</span>",
                       x=0.5, y=0.5, showarrow=False, font_size=16)
    fig.update_layout(**DARK_LAYOUT, title="CO₂ Share by Route", height=320)
    c3, c4 = st.columns(2)
    c3.plotly_chart(fig, use_container_width=True)

    # Eco rank table
    df = pd.DataFrame([
        {"Route": f"R{m['route_id']} {m['vehicle_name']}",
         "CO₂/km": round(m["co2_kg"]/max(m["distance_km"],0.01), 4),
         "CO₂/delivery": round(m["co2_kg"]/max(m["num_deliveries"],1), 4),
         "Vehicle Type": m["vehicle_type"],
         "Eco Score": round(100 - m["co2_kg"]/max(total_co2,0.01)*100, 1)}
        for m in metrics
    ]).sort_values("CO₂/km")
    c4.markdown("##### Eco-Efficiency Ranking")
    c4.dataframe(df, use_container_width=True, hide_index=True)

    # Scenario comparison: electric fleet
    st.markdown("##### 🔋 What-If: Electric Fleet Scenario")
    elec_co2 = sum(m["distance_km"] * 0.055 for m in metrics)
    diesel_co2 = total_co2
    saving_pct = (diesel_co2 - elec_co2) / max(diesel_co2, 0.001) * 100
    c5, c6, c7 = st.columns(3)
    c5.metric("Current Fleet CO₂", f"{diesel_co2:.2f} kg")
    c6.metric("Electric Fleet CO₂", f"{elec_co2:.2f} kg")
    c7.metric("Potential Saving", f"{saving_pct:.1f}%", delta=f"-{diesel_co2-elec_co2:.2f} kg")


def tab_traffic_signals(metrics, all_locs, dist_matrix):
    st.subheader("🚦 Traffic & Signal Management")

    c1, c2 = st.columns(2)
    c1.plotly_chart(chart_traffic_impact(metrics, st.session_state.traffic_factors),
                    use_container_width=True)
    c2.plotly_chart(chart_vehicle_comparison(), use_container_width=True)

    if not metrics:
        st.info("Run the solver first to see signal analysis.")
        return

    st.markdown("##### Green-Wave Signal Analysis")
    route_names = [f"Route {m['route_id']}: {m['vehicle_name']}" for m in metrics]
    sel = st.selectbox("Analyse route", route_names, key="sig_route_sel")
    sel_idx = route_names.index(sel)
    met = metrics[sel_idx]

    sigs = calculate_signal_timing(
        met["route_indices"], all_locs, dist_matrix, st.session_state.cycle_time
    )

    if sigs:
        c3, c4 = st.columns(2)
        gantt = chart_signal_gantt(sigs)
        if gantt: c3.plotly_chart(gantt, use_container_width=True)
        align = chart_alignment_scores(sigs)
        if align: c4.plotly_chart(align, use_container_width=True)

        # Summary metrics
        green_count = sum(1 for s in sigs if s["green_wave_ok"])
        avg_align   = np.mean([s["alignment_score"] for s in sigs])
        total_delay = sum(s["delay_estimate_min"] for s in sigs)
        total_stops = sum(s["expected_stops"] for s in sigs)

        cc = st.columns(4)
        cc[0].metric("🟢 Green Wave Segments", f"{green_count}/{len(sigs)}")
        cc[1].metric("📊 Avg Alignment Score",  f"{avg_align:.0f}%")
        cc[2].metric("🛑 Expected Signal Stops",str(total_stops))
        cc[3].metric("⏱ Est. Signal Delay",     f"{total_delay:.1f} min")

        # Table
        df = pd.DataFrame(sigs)[["segment","distance_km","travel_time_min",
                                   "num_signals","alignment_score","green_wave_ok",
                                   "expected_stops","delay_estimate_min"]]
        df.columns = ["Segment","Dist(km)","Time(min)","Signals","Align%","GreenWave","StopEst","Delay(min)"]
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Optimisation tips
    with st.expander("💡 Green-Wave Optimisation Tips"):
        st.markdown("""
        **How Green-Wave Timing Works:**
        - Signals along a corridor are offset so a vehicle travelling at the design speed hits every green.
        - Our alignment score measures how well the current route timing matches the signal phases.
        - **Score ≥ 70%** → Route is well-aligned; few stops expected.
        - **Score < 40%** → Consider delaying departure by half a cycle (≈45 sec) to re-align.

        **Recommendations:**
        1. Prioritise routes through South & Central zones in mid-day (lower traffic multiplier).
        2. Schedule early-morning departures for North/East routes to catch pre-rush green phases.
        3. Use electric bikes for last-mile in dense South Mumbai (lower CO₂, easier to stop/start).
        """)


def tab_advanced_analytics(metrics, all_locs, dist_matrix):
    st.subheader("📈 Advanced Analytics")

    if dist_matrix is not None:
        st.plotly_chart(chart_distance_heatmap(dist_matrix, all_locs), use_container_width=True)

    if not metrics:
        st.info("Run the solver to see Pareto and demand analysis.")
        return

    c1, c2 = st.columns(2)
    c1.plotly_chart(chart_pareto(metrics),              use_container_width=True)
    c2.plotly_chart(chart_demand_pie(metrics, all_locs),use_container_width=True)

    # Shadow prices (approximate marginal cost of adding 1 unit demand to each route)
    st.markdown("##### Shadow Prices (Marginal Delivery Cost ₹/unit)")
    shadow = []
    for m in metrics:
        spare = m["capacity"] - m["load"]
        avg_leg = m["distance_km"] / max(m["num_deliveries"], 1)
        marginal = avg_leg * m["cost_per_km"] if spare > 0 else float("inf")
        shadow.append({"Route": f"R{m['route_id']} {m['vehicle_name']}",
                        "Spare Capacity": spare,
                        "Marginal Cost (₹/unit)": round(marginal, 2) if spare > 0 else "Full"})
    st.dataframe(pd.DataFrame(shadow), use_container_width=True, hide_index=True)

    # Cumulative load chart
    st.markdown("##### Cumulative Load Along Each Route")
    fig = go.Figure()
    for i, m in enumerate(metrics):
        cum_loads = []
        total = 0.0
        for dem in m["route_demands"]:
            total += dem
            cum_loads.append(total)
        fig.add_trace(go.Scatter(
            x=list(range(len(cum_loads))), y=cum_loads,
            mode="lines+markers", name=f"R{m['route_id']}",
            line=dict(color=ROUTE_COLORS[i % len(ROUTE_COLORS)], width=2),
        ))
        fig.add_hline(y=m["capacity"], line_dash="dot",
                      line_color=ROUTE_COLORS[i % len(ROUTE_COLORS)],
                      annotation_text=f"Cap R{m['route_id']}")
    fig.update_layout(**DARK_LAYOUT, title="Cumulative Demand per Stop",
                      xaxis_title="Stop #", yaxis_title="Cumulative Load (units)", height=340)
    st.plotly_chart(fig, use_container_width=True)


def tab_delivery_plan(metrics, all_locs):
    st.subheader("📋 Delivery Plan & Recommendations")
    if not metrics:
        st.info("Run the solver to generate delivery plan.")
        return

    # Priority-based recommendations
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    delivery_locs  = [l for l in all_locs if l["type"] == "delivery"]

    rec_rows = []
    for met in metrics:
        cum_time = 0.0
        for j, (stop, coord) in enumerate(zip(met["stops"], met["coords"])):
            if j == 0 or j == len(met["stops"]) - 1:
                continue
            # Find priority
            loc_info = next((l for l in delivery_locs if l["name"] == stop), {})
            priority = loc_info.get("priority", "Medium")
            demand   = met["route_demands"][j]

            # Estimate ETA assuming 30 km/h average + 5 min unload per stop
            cum_time += 5  # rough time per stop in minutes
            rec_rows.append({
                "Route":     f"R{met['route_id']}",
                "Vehicle":   met["vehicle_name"],
                "Stop #":    j,
                "Location":  stop,
                "Zone":      loc_info.get("zone", "?"),
                "Priority":  priority,
                "Demand":    demand,
                "ETA (min)": round(cum_time, 0),
                "Rec Action":("⚡ Dispatch First" if priority == "High"
                              else "🕐 Schedule Normally" if priority == "Medium"
                              else "📆 Batch if Possible"),
            })
    df = pd.DataFrame(rec_rows).sort_values(
        ["Route", "Priority"],
        key=lambda c: c.map(priority_order) if c.name == "Priority" else c
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Export
    csv = df.to_csv(index=False)
    st.download_button("⬇️ Download Plan (CSV)", csv,
                       "delivery_plan.csv", "text/csv", use_container_width=True)

    # Stats per zone
    st.markdown("##### Deliveries by Zone")
    zone_stats = df.groupby("Zone").agg(
        Deliveries=("Location","count"), TotalDemand=("Demand","sum")
    ).reset_index()
    fig = go.Figure(go.Bar(x=zone_stats["Zone"], y=zone_stats["TotalDemand"],
                            text=zone_stats["Deliveries"],
                            texttemplate="<b>%{text} stops</b>",
                            marker_color=_colors_for(len(zone_stats))))
    fig.update_layout(**DARK_LAYOUT, title="Total Demand by Zone", height=300)
    st.plotly_chart(fig, use_container_width=True)


def tab_comparison(baseline, metrics_h, metrics_o):
    st.subheader("⚡ Solver & Scenario Comparison")
    if baseline is None:
        st.info("Run the solver at least once to see comparison.")
        return

    st.plotly_chart(chart_comparison(baseline, metrics_h, metrics_o), use_container_width=True)

    # Savings table
    rows = [{"Scenario": "Baseline (Direct)", "Cost ₹": baseline["baseline_cost_inr"],
             "CO₂ kg": baseline["baseline_co2_kg"], "Dist km": baseline["total_distance_km"],
             "Saving vs Baseline": "—"}]
    if metrics_h:
        tc = sum(m["cost_inr"] for m in metrics_h)
        td = sum(m["distance_km"] for m in metrics_h)
        tco2 = sum(m["co2_kg"] for m in metrics_h)
        rows.append({"Scenario": "Heuristic (NN+2-opt)", "Cost ₹": tc,
                     "CO₂ kg": tco2, "Dist km": td,
                     "Saving vs Baseline": f"₹{baseline['baseline_cost_inr']-tc:,.0f} ({(baseline['baseline_cost_inr']-tc)/max(baseline['baseline_cost_inr'],1)*100:.0f}%)"})
    if metrics_o:
        tc = sum(m["cost_inr"] for m in metrics_o)
        td = sum(m["distance_km"] for m in metrics_o)
        tco2 = sum(m["co2_kg"] for m in metrics_o)
        rows.append({"Scenario": "OR-Tools (GLS)", "Cost ₹": tc,
                     "CO₂ kg": tco2, "Dist km": td,
                     "Saving vs Baseline": f"₹{baseline['baseline_cost_inr']-tc:,.0f} ({(baseline['baseline_cost_inr']-tc)/max(baseline['baseline_cost_inr'],1)*100:.0f}%)"})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # No-traffic scenario
    if metrics_h:
        st.markdown("##### Traffic Impact Simulation")
        baseline_tf = {"North":1.0,"Central":1.0,"South":1.0,"East":1.0}
        dm   = st.session_state.dist_matrix
        al   = st.session_state.all_locations
        dem  = st.session_state.demands
        veh  = st.session_state.vehicles
        zones= [l.get("zone","Central") for l in al]
        no_traffic_time = apply_traffic_multipliers(dm, zones, baseline_tf)
        no_traffic_m = calculate_route_metrics(
            st.session_state.routes, al, dm, no_traffic_time, dem, veh
        )

        current_time = sum(m["travel_time_hr"] for m in metrics_h)
        no_tf_time   = sum(m["travel_time_hr"] for m in no_traffic_m)
        extra_hr = current_time - no_tf_time
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Free-Flow", x=[m["vehicle_name"] for m in no_traffic_m],
                               y=[m["travel_time_hr"] for m in no_traffic_m], marker_color="#2ECC71"))
        fig2.add_trace(go.Bar(name="With Traffic", x=[m["vehicle_name"] for m in metrics_h],
                               y=[m["travel_time_hr"] for m in metrics_h], marker_color="#E74C3C"))
        fig2.update_layout(**DARK_LAYOUT, barmode="group",
                            title=f"Travel Time: Free-Flow vs With Traffic (+{extra_hr:.1f} hr extra)",
                            yaxis_title="Travel Time (hr)", height=340)
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # Hero banner
    st.markdown(
        """
        <div class="hero-banner">
          <h1>🚚 Urban Logistics Optimizer — Mumbai</h1>
          <p>Vehicle Routing Problem (CVRP) · OR-Tools + Heuristic · Real-time Traffic · CO₂ Tracking</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics    = st.session_state.metrics
    all_locs   = st.session_state.all_locations or (
        [{**w, "type":"warehouse","demand":0} for w in st.session_state.warehouses] +
        [{**d, "type":"delivery"} for d in st.session_state.delivery_points]
    )
    dist_mat   = st.session_state.dist_matrix
    baseline   = st.session_state.baseline

    # KPI strip
    if metrics and baseline:
        render_kpi_row(metrics, baseline)
        st.markdown(
            f"<p style='color:#888;font-size:.8rem;margin-top:6px'>"
            f"Solved with <b>{st.session_state.solver_used}</b> · "
            f"{len(metrics)} routes · "
            f"{sum(m['num_deliveries'] for m in metrics)} deliveries</p>",
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "👈 **Configure** warehouses, fleet, and delivery points in the sidebar, "
            "then click **🚀 Solve VRP** to optimise routes."
        )

    st.divider()

    # Tabs
    tabs = st.tabs([
        "🗺️ Route Map",
        "📊 Route Analysis",
        "🌿 Emissions",
        "🚦 Traffic & Signals",
        "📈 Advanced Analytics",
        "📋 Delivery Plan",
        "⚡ Comparison",
    ])

    with tabs[0]: tab_route_map(metrics, all_locs)
    with tabs[1]: tab_route_analysis(metrics)
    with tabs[2]: tab_emissions(metrics)
    with tabs[3]: tab_traffic_signals(metrics, all_locs,
                                      dist_mat if dist_mat is not None
                                      else build_distance_matrix(all_locs))
    with tabs[4]: tab_advanced_analytics(metrics, all_locs,
                                          dist_mat if dist_mat is not None
                                          else build_distance_matrix(all_locs))
    with tabs[5]: tab_delivery_plan(metrics, all_locs)
    with tabs[6]: tab_comparison(baseline,
                                  st.session_state.heuristic_metrics,
                                  st.session_state.ortools_metrics)

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='color:#555;font-size:.75rem;text-align:center'>"
        "Mumbai Urban Logistics Optimizer · OR-Tools CVRP + NN Heuristic · "
        "Haversine distances · Streamlit Cloud deployable</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()