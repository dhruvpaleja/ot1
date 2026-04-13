"""
vrp_engine.py — VRP Solver Engine for Urban Logistics Optimization
Supports:
  • OR-Tools CVRP (with GLS metaheuristic)
  • Nearest-Neighbor heuristic + 2-opt improvement
  • Traffic-aware travel-time matrix
  • Signal timing / green-wave analysis
  • Full route metrics & baseline comparison
"""

from __future__ import annotations
import numpy as np
from math import radians, cos, sin, asin, sqrt
from typing import List, Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────
# 1.  GEOMETRY
# ─────────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two WGS-84 coordinates."""
    R = 6_371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * R * asin(sqrt(max(0.0, a)))


def build_distance_matrix(locations: List[Dict]) -> np.ndarray:
    """
    Build an (n×n) symmetric distance matrix (km) using Haversine ×
    an urban-road factor (1.35) to account for non-straight roads.
    """
    n = len(locations)
    road_factor = 1.35
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i, j] = (
                    haversine(
                        locations[i]["lat"], locations[i]["lon"],
                        locations[j]["lat"], locations[j]["lon"]
                    )
                    * road_factor
                )
    return mat


# ─────────────────────────────────────────────────────────────
# 2.  TRAFFIC
# ─────────────────────────────────────────────────────────────

def apply_traffic_multipliers(
    dist_matrix: np.ndarray,
    location_zones: List[str],
    traffic_factors: Dict[str, float],
    avg_speed_kmh: float = 30.0,
) -> np.ndarray:
    """
    Convert a distance matrix (km) → travel-time matrix (hours)
    with per-zone traffic multipliers applied symmetrically to each arc.
    """
    n = dist_matrix.shape[0]
    time_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                tf = (
                    traffic_factors.get(location_zones[i], 1.0)
                    + traffic_factors.get(location_zones[j], 1.0)
                ) / 2.0
                time_mat[i, j] = (dist_matrix[i, j] / avg_speed_kmh) * tf
    return time_mat


# ─────────────────────────────────────────────────────────────
# 3.  HEURISTIC SOLVER  (Nearest Neighbour + 2-opt)
# ─────────────────────────────────────────────────────────────

def nearest_neighbor_solver(
    dist_matrix: np.ndarray,
    demands: List[float],
    vehicle_capacity: float,
    num_vehicles: int,
    depot_idx: int = 0,
) -> List[List[int]]:
    """
    Greedy nearest-neighbour construction heuristic for CVRP.
    Returns a list of routes; each route is a list of node indices
    starting and ending at *depot_idx*.
    """
    n = dist_matrix.shape[0]
    customers = [i for i in range(n) if i != depot_idx]
    unvisited = set(customers)
    routes: List[List[int]] = []

    for _ in range(num_vehicles):
        if not unvisited:
            break
        route = [depot_idx]
        load = 0.0
        current = depot_idx

        while unvisited:
            best_d, best_node = float("inf"), None
            for node in sorted(unvisited):
                if demands[node] > 0 and load + demands[node] <= vehicle_capacity:
                    d = dist_matrix[current, node]
                    if d < best_d:
                        best_d, best_node = d, node
            if best_node is None:
                break
            route.append(best_node)
            load += demands[best_node]
            unvisited.remove(best_node)
            current = best_node

        route.append(depot_idx)
        if len(route) > 2:
            routes.append(route)

    return routes


def _route_cost(route: List[int], dist_matrix: np.ndarray) -> float:
    return sum(dist_matrix[route[k], route[k + 1]] for k in range(len(route) - 1))


def two_opt_route(route: List[int], dist_matrix: np.ndarray) -> List[int]:
    """Improve a single route with 2-opt edge-swaps."""
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if _route_cost(candidate, dist_matrix) < _route_cost(best, dist_matrix) - 1e-9:
                    best = candidate
                    improved = True
    return best


def two_opt_improve(routes: List[List[int]], dist_matrix: np.ndarray) -> List[List[int]]:
    return [two_opt_route(r, dist_matrix) for r in routes]


# ─────────────────────────────────────────────────────────────
# 4.  OR-TOOLS SOLVER
# ─────────────────────────────────────────────────────────────

def solve_with_ortools(
    dist_matrix: np.ndarray,
    demands: List[float],
    vehicle_capacity: float,
    num_vehicles: int,
    depot_idx: int = 0,
    time_limit_sec: int = 30,
) -> Tuple[Optional[List[List[int]]], str]:
    """
    Solve CVRP with Google OR-Tools (PATH_CHEAPEST_ARC + GLS).
    Returns (routes, status_string).
    """
    try:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    except ImportError:
        return None, "OR-Tools not installed — pip install ortools"

    n = dist_matrix.shape[0]
    SCALE = 1_000
    int_mat = (dist_matrix * SCALE).astype(int).tolist()
    int_dem = [int(round(d)) for d in demands]
    int_cap = int(round(vehicle_capacity))

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(fi, ti):
        return int_mat[manager.IndexToNode(fi)][manager.IndexToNode(ti)]

    tc = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(tc)

    def dem_cb(fi):
        return int_dem[manager.IndexToNode(fi)]

    dc = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimensionWithVehicleCapacity(
        dc, 0, [int_cap] * num_vehicles, True, "Capacity"
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = time_limit_sec

    sol = routing.SolveWithParameters(params)
    if not sol:
        return None, "OR-Tools: no feasible solution found"

    routes: List[List[int]] = []
    for v in range(num_vehicles):
        idx = routing.Start(v)
        route: List[int] = []
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        route.append(manager.IndexToNode(idx))
        if len(route) > 2:
            routes.append(route)

    total_dist = sum(
        sum(dist_matrix[r[k], r[k + 1]] for k in range(len(r) - 1))
        for r in routes
    )
    return routes, f"OR-Tools OK — {len(routes)} routes, {total_dist:.1f} km total"


# ─────────────────────────────────────────────────────────────
# 5.  METRICS
# ─────────────────────────────────────────────────────────────

def calculate_route_metrics(
    routes: List[List[int]],
    locations: List[Dict],
    dist_matrix: np.ndarray,
    time_matrix: np.ndarray,
    demands: List[float],
    vehicles: List[Dict],
) -> List[Dict]:
    """
    For every route compute distance, time, load, cost (₹), CO₂ (kg)
    and detailed stop info.
    """
    metrics: List[Dict] = []
    for idx, route in enumerate(routes):
        v = vehicles[idx % len(vehicles)]
        dist   = sum(dist_matrix[route[k], route[k + 1]] for k in range(len(route) - 1))
        t_hr   = sum(time_matrix[route[k], route[k + 1]] for k in range(len(route) - 1))
        load   = sum(demands[node] for node in route[1:-1])
        cost   = dist * v["cost_per_km"]

        # CO₂: base rate × distance × load factor × stop-and-go penalty
        load_factor   = 1.0 + 0.15 * (load / max(v["capacity"], 1))
        traffic_ratio = (t_hr / (dist / 30)) if dist > 0 else 1.0
        sng_penalty   = 1.0 + max(0.0, 0.10 * (traffic_ratio - 1.0))
        co2 = dist * v["co2_per_km"] * load_factor * sng_penalty

        stops  = [locations[n]["name"] for n in route]
        coords = [{"lat": locations[n]["lat"], "lon": locations[n]["lon"]} for n in route]
        route_demands = [demands[n] for n in route]

        metrics.append(
            dict(
                route_id       = idx + 1,
                vehicle_name   = v["name"],
                vehicle_type   = v["type"],
                stops          = stops,
                coords         = coords,
                route_indices  = route,
                route_demands  = route_demands,
                distance_km    = round(dist, 2),
                travel_time_hr = round(t_hr, 2),
                load           = round(load, 1),
                capacity       = v["capacity"],
                utilization_pct= round(load / v["capacity"] * 100, 1) if v["capacity"] else 0,
                cost_inr       = round(cost, 2),
                co2_kg         = round(co2, 3),
                num_deliveries = len(route) - 2,
                cost_per_km    = v["cost_per_km"],
                co2_per_km     = v["co2_per_km"],
                load_factor    = round(load_factor, 3),
                traffic_ratio  = round(traffic_ratio, 3),
            )
        )
    return metrics


def compute_baseline_metrics(
    locations: List[Dict],
    dist_matrix: np.ndarray,
    time_matrix: np.ndarray,
    demands: List[float],
    depot_idx: int,
) -> Dict:
    """
    Naive baseline: each delivery served by a direct round-trip from depot.
    Uses a standard diesel van (₹15/km, 0.27 kg CO₂/km).
    """
    customers = [i for i in range(len(locations)) if i != depot_idx]
    total_dist = sum(dist_matrix[depot_idx, c] * 2 for c in customers)
    total_time = sum(time_matrix[depot_idx, c] * 2 for c in customers)
    return dict(
        total_distance_km   = round(total_dist, 2),
        total_time_hr       = round(total_time, 2),
        total_demand        = round(sum(demands[c] for c in customers), 1),
        baseline_co2_kg     = round(total_dist * 0.27, 2),
        baseline_cost_inr   = round(total_dist * 15.0, 2),
        num_customers       = len(customers),
    )


# ─────────────────────────────────────────────────────────────
# 6.  SIGNAL TIMING  (green-wave analysis)
# ─────────────────────────────────────────────────────────────

def calculate_signal_timing(
    route: List[int],
    locations: List[Dict],
    dist_matrix: np.ndarray,
    cycle_time: int = 90,
    avg_speed_kmh: float = 30.0,
) -> List[Dict]:
    """
    Simulate traffic-signal phases along a route for green-wave
    optimisation.  Returns one record per road segment.
    """
    if len(route) < 2:
        return []

    records: List[Dict] = []
    cum_min = 0.0

    for k in range(len(route) - 1):
        fn, tn   = route[k], route[k + 1]
        dist_km  = dist_matrix[fn, tn]
        trav_min = (dist_km / avg_speed_kmh) * 60.0

        # Signals along segment: ~1 per 400 m urban
        n_signals   = max(1, int(dist_km / 0.4))
        arrival_min = cum_min + trav_min
        phase_sec   = (arrival_min * 60) % cycle_time        # seconds into cycle
        green_sec   = cycle_time * 0.45
        red_sec     = cycle_time - green_sec

        # Alignment score: how close is arrival to the middle of green phase
        best_phase  = green_sec / 2.0
        align_score = max(0.0, 100.0 - abs(phase_sec - best_phase) / (cycle_time / 2.0) * 100.0)

        # Expected signal stops along segment
        red_ratio     = red_sec / cycle_time
        expected_stops = max(0, round(n_signals * red_ratio * (1 - align_score / 100)))

        records.append(dict(
            segment          = f"{locations[fn]['name']} → {locations[tn]['name']}",
            from_name        = locations[fn]["name"],
            to_name          = locations[tn]["name"],
            distance_km      = round(dist_km, 2),
            travel_time_min  = round(trav_min, 1),
            num_signals      = n_signals,
            arrival_time_min = round(arrival_min, 1),
            phase_offset_sec = round(phase_sec, 1),
            green_time_sec   = round(green_sec, 1),
            red_time_sec     = round(red_sec, 1),
            cycle_time_sec   = cycle_time,
            alignment_score  = round(align_score, 1),
            green_wave_ok    = phase_sec <= green_sec,
            expected_stops   = expected_stops,
            delay_estimate_min = round(expected_stops * (red_sec / 60) * 0.5, 2),
        ))
        cum_min = arrival_min

    return records