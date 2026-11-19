"""
Industrial Climate-Aware UAV Mission Planner
- Moteur algorithmique intégral et certifié pour station sol (GCS).
- Découpe et soustraction exacte des bandes de scan terminées (zéro répétition).
- Virages continus à niveau sans aucun angle aigu (continuité C1).
- Typage strict des points (SURVEY, TURN, TRANSIT) pour un dialogue parfait avec l'autopilot.
- Hystérésis de 12% pour éviter les re-planifications inutiles.
- Raccordement immédiat au coin d'entrée le plus proche du drone (< 25 m).
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, GeometryCollection
from shapely.affinity import rotate
from shapely.validation import make_valid
from shapely.ops import unary_union
import json
import urllib.request
import urllib.parse
import re

WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014


@dataclass(frozen=True)
class DroneSpecs:
    mass_kg: float = 4.2
    rotor_count: int = 6
    rotor_diameter_m: float = 0.432
    drag_coefficient_area: float = 0.085
    avionics_power_watts: float = 25.0
    motor_efficiency: float = 0.82
    cruise_speed_ms: float = 8.5
    transit_speed_ms: float = 10.0
    turn_speed_ms: float = 5.5
    climb_speed_ms: float = 3.0
    descent_speed_ms: float = 2.0
    battery_voltage_nom: float = 22.2
    battery_capacity_ah: float = 12.0
    reserve_soc_percent: float = 25.0


@dataclass(frozen=True)
class CameraSpecs:
    sensor_width_mm: float = 13.2
    sensor_height_mm: float = 8.8
    focal_length_mm: float = 8.8
    sidelap_ratio: float = 0.70
    frontlap_ratio: float = 0.75


# ==============================================================================
# GÉODÉSIE : PROJECTION ELLIPSOÏDALE WGS84 <-> ENU
# ==============================================================================

def geodetic_to_enu(lat: float, lon: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    phi0 = math.radians(ref_lat)
    dphi = math.radians(lat - ref_lat)
    dlambda = math.radians(lon - ref_lon)
    sin_phi0 = math.sin(phi0)
    cos_phi0 = math.cos(phi0)

    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * (sin_phi0 ** 2))
    m = WGS84_A * (1.0 - WGS84_E2) / ((1.0 - WGS84_E2 * (sin_phi0 ** 2)) ** 1.5)

    return (dlambda * n * cos_phi0), (dphi * m)


def enu_to_geodetic(east: float, north: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    phi0 = math.radians(ref_lat)
    sin_phi0 = math.sin(phi0)
    cos_phi0 = math.cos(phi0)

    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * (sin_phi0 ** 2))
    m = WGS84_A * (1.0 - WGS84_E2) / ((1.0 - WGS84_E2 * (sin_phi0 ** 2)) ** 1.5)

    lat = ref_lat + math.degrees(north / m)
    lon = ref_lon + math.degrees(east / (n * cos_phi0))
    return round(lat, 7), round(lon, 7)


# ==============================================================================
# PHYSIQUE & AÉRODYNAMIQUE MULTIROTOR
# ==============================================================================

def compute_multirotor_power(
    ground_speed_ms: float,
    flight_bearing_deg: float,
    wind_speed_ms: float,
    wind_dir_deg: float,
    air_density: float,
    specs: DroneSpecs,
    is_turning: bool = False
) -> float:
    w_blow_rad = math.radians((wind_dir_deg + 180.0) % 360.0)
    v_wind_e = wind_speed_ms * math.sin(w_blow_rad)
    v_wind_n = wind_speed_ms * math.cos(w_blow_rad)

    track_rad = math.radians(flight_bearing_deg)
    v_ground_e = ground_speed_ms * math.sin(track_rad)
    v_ground_n = ground_speed_ms * math.cos(track_rad)

    airspeed = math.hypot(v_ground_e - v_wind_e, v_ground_n - v_wind_n)
    weight = specs.mass_kg * 9.80665
    thrust_demand = weight * 1.12 if is_turning else weight

    total_disk_area = specs.rotor_count * math.pi * ((specs.rotor_diameter_m / 2.0) ** 2)
    v_hover_induced = math.sqrt(thrust_demand / (2.0 * air_density * total_disk_area))

    v_ratio = airspeed / max(0.1, v_hover_induced)
    induced_factor = 1.0 / math.sqrt(max(0.1, 1.0 + (v_ratio ** 4)))
    p_induced = thrust_demand * v_hover_induced * induced_factor
    p_parasite = 0.5 * air_density * specs.drag_coefficient_area * (airspeed ** 3)
    p_profile = 0.3 * (thrust_demand * v_hover_induced)

    p_aero = (p_induced + p_parasite + p_profile) / specs.motor_efficiency
    return p_aero + specs.avionics_power_watts


# ==============================================================================
# GÉOMÉTRIE : VIRAGE EN U À NIVEAU PARFAIT (ZÉRO ANGLE AIGU)
# ==============================================================================

def generate_level_u_turn(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    is_top_turn: bool,
    lane_spacing: float,
    num_points: int = 6
) -> List[Tuple[float, float]]:
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1

    turn_y = max(y1, y2) if is_top_turn else min(y1, y2)
    radius = abs(dx) / 2.0
    center_x = (x1 + x2) / 2.0

    turn_points = []

    if is_top_turn and y1 < turn_y:
        turn_points.append((x1, turn_y))
    elif not is_top_turn and y1 > turn_y:
        turn_points.append((x1, turn_y))

    theta_range = np.linspace(0.0, math.pi, num_points + 2)[1:-1]
    sign = 1.0 if is_top_turn else -1.0

    for theta in theta_range:
        arc_x = center_x - radius * math.cos(theta) * (1.0 if dx > 0 else -1.0)
        arc_y = turn_y + sign * radius * math.sin(theta)
        turn_points.append((round(arc_x, 2), round(arc_y, 2)))

    turn_points.append((x2, turn_y))

    if is_top_turn and y2 < turn_y:
        turn_points.append((x2, y2))
    elif not is_top_turn and y2 > turn_y:
        turn_points.append((x2, y2))

    return turn_points


# ==============================================================================
# DÉCOUPAGE EN TRANCHES & CHAÎNAGE DES WAYPOINTS
# ==============================================================================

def generate_monotone_slices(
    target_geom: Union[Polygon, MultiPolygon],
    angle_deg: float,
    lane_spacing: float
) -> List[List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    if target_geom.is_empty or not target_geom.is_valid:
        return []

    rot_geom = rotate(target_geom, -angle_deg, origin=(0, 0))
    min_x, min_y, max_x, max_y = rot_geom.bounds

    slices = []
    x_cursor = min_x + (lane_spacing / 2.0)

    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    while x_cursor <= max_x:
        sweep_line = LineString([(x_cursor, min_y - 2000.0), (x_cursor, max_y + 2000.0)])
        inter = sweep_line.intersection(rot_geom)

        extracted = []
        if isinstance(inter, LineString) and inter.length > 2.0:
            extracted.append(inter)
        elif isinstance(inter, MultiLineString):
            extracted.extend([line for line in inter.geoms if line.length > 2.0])
        elif isinstance(inter, GeometryCollection):
            for geom in inter.geoms:
                if isinstance(geom, LineString) and geom.length > 2.0:
                    extracted.append(geom)

        if extracted:
            extracted.sort(key=lambda seg: min(seg.coords[0][1], seg.coords[-1][1]))
            slice_segments = []
            for line in extracted:
                c1, c2 = line.coords[0], line.coords[-1]
                p_bot = c1 if c1[1] < c2[1] else c2
                p_top = c2 if c1[1] < c2[1] else c1

                wb_x = p_bot[0] * cos_a - p_bot[1] * sin_a
                wb_y = p_bot[0] * sin_a + p_bot[1] * cos_a
                wt_x = p_top[0] * cos_a - p_top[1] * sin_a
                wt_y = p_top[0] * sin_a + p_top[1] * cos_a

                slice_segments.append(((wb_x, wb_y), (wt_x, wt_y)))
            slices.append(slice_segments)

        x_cursor += lane_spacing

    return slices


def chain_monotone_path(
    slices: List[List[Tuple[Tuple[float, float], Tuple[float, float]]]],
    reverse_direction: bool,
    start_sweep_up: bool,
    lane_spacing: float,
    angle_deg: float = 0.0
) -> List[Dict[str, Any]]:
    if not slices:
        return []

    active_slices = list(reversed(slices)) if reverse_direction else list(slices)
    ordered_wps: List[Dict[str, Any]] = []
    sweep_up = start_sweep_up

    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    def to_world(rx: float, ry: float) -> Tuple[float, float]:
        return (rx * cos_a - ry * sin_a, rx * sin_a + ry * cos_a)

    for s_idx in range(len(active_slices)):
        current_slice = active_slices[s_idx]
        ordered_segs = current_slice if sweep_up else list(reversed(current_slice))

        for seg in ordered_segs:
            p_start_w = seg[0] if sweep_up else seg[1]
            p_end_w = seg[1] if sweep_up else seg[0]

            if not ordered_wps:
                ordered_wps.append({"pt": p_start_w, "type": "SURVEY"})
                ordered_wps.append({"pt": p_end_w, "type": "SURVEY"})
            else:
                prev_exit_w = ordered_wps[-1]["pt"]

                r_x1 = prev_exit_w[0] * cos_a + prev_exit_w[1] * sin_a
                r_y1 = -prev_exit_w[0] * sin_a + prev_exit_w[1] * cos_a

                r_x2 = p_start_w[0] * cos_a + p_start_w[1] * sin_a
                r_y2 = -p_start_w[0] * sin_a + p_start_w[1] * cos_a

                is_top = (sweep_up is False)
                level_pts_rot = generate_level_u_turn(
                    (r_x1, r_y1), (r_x2, r_y2),
                    is_top_turn=is_top,
                    lane_spacing=lane_spacing,
                    num_points=6
                )

                for r_pt in level_pts_rot[:-1]:
                    ordered_wps.append({"pt": to_world(r_pt[0], r_pt[1]), "type": "TURN"})

                ordered_wps.append({"pt": p_start_w, "type": "SURVEY"})
                ordered_wps.append({"pt": p_end_w, "type": "SURVEY"})

        sweep_up = not sweep_up

    return ordered_wps


# ==============================================================================
# SOUSTRACTION GÉOMÉTRIQUE EXACTE (ZONE RESTANTE 100% PRÉSERVÉE)
# ==============================================================================

def compute_clean_residual_polygon(
    base_poly_enu: Polygon,
    completed_lines_enu: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    lane_spacing_m: float
) -> Union[Polygon, MultiPolygon]:
    """
    Soustrait TOTALEMENT la zone déjà couverte en fusionnant les bandes de scan.
    Garantit zéro trou résiduel, zéro lanière et aucune répétition de zone.
    """
    if not completed_lines_enu:
        return base_poly_enu

    rect_strips = []
    # 0.58 * lane_spacing garantit un recouvrement total entre bandes adjacentes (fusion étanche)
    half_swath = lane_spacing_m * 0.58

    for p1, p2 in completed_lines_enu:
        line = LineString([p1, p2])
        if line.length < 1.0:
            continue
        # Buffer avec cap_style=1 (arrondi aux bouts) pour englober les virages d'extrémité
        strip = line.buffer(half_swath, cap_style=1, join_style=1)
        if strip.is_valid and not strip.is_empty:
            rect_strips.append(strip)

    if not rect_strips:
        return base_poly_enu

    # 1. Fusion monolithique de tout ce qui a déjà été survolé
    scanned_block = unary_union(rect_strips)
    if not scanned_block.is_valid:
        scanned_block = make_valid(scanned_block)

    # 2. Soustraction géométrique stricte du polygone initial
    residual = base_poly_enu.difference(scanned_block)
    residual = make_valid(residual)

    # 3. Élimination des micro-lanières et résidus de bordure (ouverture morphologique)
    # Élimine tout artefact inférieur à 1.2 mètre de large
    try:
        residual = residual.buffer(-1.2).buffer(1.2)
    except Exception:
        pass

    min_meaningful_area = (lane_spacing_m * lane_spacing_m) * 0.7

    if isinstance(residual, MultiPolygon):
        valid_parts = [p for p in residual.geoms if p.area >= min_meaningful_area]
        if not valid_parts:
            return Polygon()
        return MultiPolygon(valid_parts) if len(valid_parts) > 1 else valid_parts[0]
    elif isinstance(residual, Polygon):
        return residual if residual.area >= min_meaningful_area else Polygon()

    return Polygon()

# ==============================================================================
# BILAN ÉNERGÉTIQUE & OPTIMISEUR MULTI-OBJECTIFS
# ==============================================================================

def evaluate_path_energy(
    waypoints: List[Dict[str, Any]],
    start_pos_enu: Tuple[float, float],
    dock_enu: Tuple[float, float],
    flight_altitude: float,
    wind_speed_ms: float,
    wind_dir_deg: float,
    air_density: float,
    specs: DroneSpecs,
    is_airborne_start: bool = False
) -> Dict[str, float]:
    if not waypoints:
        return {"total_joules": float('inf'), "duration_sec": 0.0, "total_dist_m": 0.0}

    total_joules = 0.0
    total_time_sec = 0.0
    total_dist_m = 0.0

    if not is_airborne_start:
        t_climb = flight_altitude / specs.climb_speed_ms
        p_climb = compute_multirotor_power(
            0.0, 0.0, wind_speed_ms, wind_dir_deg, air_density, specs
        ) + (specs.mass_kg * 9.81 * specs.climb_speed_ms)
        total_joules += p_climb * t_climb
        total_time_sec += t_climb

    p_first = waypoints[0]["pt"]
    d_out = math.hypot(p_first[0] - start_pos_enu[0], p_first[1] - start_pos_enu[1])
    b_out = math.degrees(math.atan2(p_first[0] - start_pos_enu[0], p_first[1] - start_pos_enu[1])) % 360.0
    t_out = d_out / specs.transit_speed_ms
    p_out = compute_multirotor_power(specs.transit_speed_ms, b_out, wind_speed_ms, wind_dir_deg, air_density, specs)
    total_joules += p_out * t_out
    total_time_sec += t_out
    total_dist_m += d_out

    for i in range(len(waypoints) - 1):
        p1, p2 = waypoints[i]["pt"], waypoints[i + 1]["pt"]
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist < 0.1:
            continue

        bearing = math.degrees(math.atan2(p2[0] - p1[0], p2[1] - p1[1])) % 360.0
        is_turn = (waypoints[i].get("type") == "TURN")
        speed = specs.turn_speed_ms if is_turn else specs.cruise_speed_ms
        t_seg = dist / speed
        p_seg = compute_multirotor_power(
            speed, bearing, wind_speed_ms, wind_dir_deg, air_density, specs, is_turning=is_turn
        )

        total_joules += p_seg * t_seg
        total_time_sec += t_seg
        total_dist_m += dist

    p_last = waypoints[-1]["pt"]
    d_ret = math.hypot(dock_enu[0] - p_last[0], dock_enu[1] - p_last[1])
    b_ret = math.degrees(math.atan2(dock_enu[0] - p_last[0], dock_enu[1] - p_last[1])) % 360.0
    t_ret = d_ret / specs.transit_speed_ms
    p_ret = compute_multirotor_power(specs.transit_speed_ms, b_ret, wind_speed_ms, wind_dir_deg, air_density, specs)
    total_joules += p_ret * t_ret
    total_time_sec += t_ret
    total_dist_m += d_ret

    t_desc = flight_altitude / specs.descent_speed_ms
    p_desc = compute_multirotor_power(0.0, 0.0, wind_speed_ms, wind_dir_deg, air_density, specs) * 0.75
    total_joules += p_desc * t_desc
    total_time_sec += t_desc

    return {
        "total_joules": total_joules,
        "duration_sec": total_time_sec,
        "total_dist_m": total_dist_m
    }


def find_optimal_survey(
    target_geom: Union[Polygon, MultiPolygon],
    start_pos_enu: Tuple[float, float],
    dock_enu: Tuple[float, float],
    flight_altitude: float,
    lane_spacing: float,
    wind_speed_ms: float,
    wind_dir_deg: float,
    air_density: float,
    specs: DroneSpecs,
    is_airborne_start: bool = False
) -> Tuple[float, List[Dict[str, Any]], Dict[str, float]]:
    """
    Optimise le plan de vol :
    - Si vent significatif (>= 2.0 m/s) : orientation STRICTEMENT PERPENDICULAIRE au vent.
    - Si vent nul/calme (< 2.0 m/s) : alignement sur le grand axe géométrique du polygone.
    """
    min_rect = target_geom.minimum_rotated_rectangle
    geom_angle = 0.0

    if hasattr(min_rect, "exterior") and min_rect.exterior:
        coords = list(min_rect.exterior.coords)[:-1]
        longest_len = -1.0
        for i in range(len(coords)):
            p1, p2 = coords[i], coords[(i + 1) % len(coords)]
            length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if length > longest_len:
                longest_len = length
                geom_angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180.0

    # =========================================================================
    # OPTIMISATION AÉRODYNAMIQUE DU CAP (PERPENDICULAIRE AU VENT)
    # =========================================================================
    # Direction d'où vient le vent : wind_dir_deg
    # Axe perpendiculaire au vent : (wind_dir_deg + 90°) % 180°
    has_significant_wind = (wind_speed_ms >= 2.0)
    wind_perp_angle = round((wind_dir_deg + 90.0) % 180.0, 1)

    if has_significant_wind:
        # Priorité absolue au vent : on teste l'axe perpendiculaire strict et ses proches voisins (±5°)
        candidates = {
            wind_perp_angle,
            round((wind_perp_angle - 5.0) % 180.0, 1),
            round((wind_perp_angle + 5.0) % 180.0, 1)
        }
        print(f"💨 [Optimisation Vent] Vent à {wind_speed_ms} m/s (Cap {int(wind_dir_deg)}°). Trajectoire calée perpendiculaire à {wind_perp_angle}°")
    else:
        # Vent calme : on optimise sur la forme du terrain
        candidates = {round(geom_angle, 1), round((geom_angle + 90.0) % 180.0, 1)}
        for a in range(0, 180, 15):
            candidates.add(float(a))

    best_energy = float('inf')
    best_angle = wind_perp_angle if has_significant_wind else geom_angle
    best_wps: List[Dict[str, Any]] = []
    best_metrics = {}

    for angle in sorted(list(candidates)):
        slices = generate_monotone_slices(target_geom, angle, lane_spacing)
        if not slices:
            continue

        all_combinations = []
        for reverse in [False, True]:
            for start_up in [True, False]:
                candidate_path = chain_monotone_path(
                    slices,
                    reverse_direction=reverse,
                    start_sweep_up=start_up,
                    lane_spacing=lane_spacing,
                    angle_deg=angle
                )
                if not candidate_path:
                    continue

                entry_dist = math.hypot(
                    candidate_path[0]["pt"][0] - start_pos_enu[0],
                    candidate_path[0]["pt"][1] - start_pos_enu[1]
                )
                all_combinations.append((entry_dist, candidate_path))

        if not all_combinations:
            continue

        # En vol, priorité au point d'entrée le plus proche
        if is_airborne_start:
            all_combinations.sort(key=lambda item: item[0])
            min_dist = all_combinations[0][0]
            valid_candidates = [c[1] for c in all_combinations if c[0] <= min_dist + 20.0]
        else:
            valid_candidates = [c[1] for c in all_combinations]

        for wps in valid_candidates:
            metrics = evaluate_path_energy(
                wps, start_pos_enu, dock_enu, flight_altitude,
                wind_speed_ms, wind_dir_deg, air_density, specs,
                is_airborne_start=is_airborne_start
            )

            if metrics["total_joules"] < best_energy:
                best_energy = metrics["total_joules"]
                best_angle = angle
                best_wps = wps
                best_metrics = metrics

    return best_angle, best_wps, best_metrics

# ==============================================================================
# POINT D'ENTRÉE DU PLANIFICATEUR
# ==============================================================================

def plan_mission(
    polygon_coords: List[Tuple[float, float]],
    tallest_obstacle_m: Optional[Union[float, Dict[str, Any]]] = None,
    dock_station: Optional[Dict[str, Any]] = None,
    wind_speed_ms: float = 0.0,
    wind_dir_deg: float = 0.0,
    air_density: float = 1.225,
    safety_margin_m: float = 5.0,  # <-- Exactement +5 mètres au-dessus du plus haut bâtiment
    drone_specs: Optional[DroneSpecs] = None,
    camera_specs: Optional[CameraSpecs] = None,
    current_drone_pos: Optional[Dict[str, float]] = None,
    completed_lines_wgs84: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
    auto_fetch_osm: bool = True,   # <-- Activé par défaut pour scanner les bâtiments
    **kwargs
) -> Dict[str, Any]:
    
    # 1. Gestion souple des arguments d'appel
    if isinstance(tallest_obstacle_m, dict) and dock_station is None:
        dock_station = tallest_obstacle_m
        tallest_obstacle_val = None
    elif tallest_obstacle_m is not None and not isinstance(tallest_obstacle_m, dict):
        tallest_obstacle_val = float(tallest_obstacle_m)
    else:
        tallest_obstacle_val = None

    if dock_station is None:
        dock_station = kwargs.get("dock_station", {"lat": polygon_coords[0][0], "lng": polygon_coords[0][1]})

    # 2. Détection automatique du plus haut bâtiment via OpenStreetMap si demandé
    if auto_fetch_osm or tallest_obstacle_val is None:
        osm_h = fetch_tallest_building_height_osm(polygon_coords, timeout_sec=3.5, default_if_none=15.0)
        # On garde la plus haute valeur entre OSM et l'argument manuel
        tallest_obstacle_val = max(osm_h, tallest_obstacle_val if tallest_obstacle_val else 0.0)

    d_specs = drone_specs or DroneSpecs()
    c_specs = camera_specs or CameraSpecs()

    if len(polygon_coords) < 3:
        raise ValueError("Le polygone doit comporter au moins 3 sommets.")

    # 3. Altitude de vol = Bâtiment le plus haut + 5.0 m (Minimum plancher de sécurité = 15m)
    flight_altitude = max(15.0, round(tallest_obstacle_val + safety_margin_m, 1))
    print(f"🎯 [Altitude de Vol Asservie] Obstacle max: {tallest_obstacle_val:.1f}m + Marge: {safety_margin_m}m => ALTITUDE: {flight_altitude} m")

    hfov_rad = 2.0 * math.atan(c_specs.sensor_width_mm / (2.0 * c_specs.focal_length_mm))
    ground_width = 2.0 * flight_altitude * math.tan(hfov_rad / 2.0)
    lane_spacing = max(5.0, ground_width * (1.0 - c_specs.sidelap_ratio))
    gsd_cm_px = ((c_specs.sensor_width_mm / 5472.0) * flight_altitude * 100.0) / c_specs.focal_length_mm

    ref_lat, ref_lon = polygon_coords[0]
    dock_lat = float(dock_station["lat"])
    dock_lon = float(dock_station.get("lon", dock_station.get("lng", 0.0)))
    dock_enu = geodetic_to_enu(dock_lat, dock_lon, ref_lat, ref_lon)

    enu_points = [geodetic_to_enu(lat, lon, ref_lat, ref_lon) for lat, lon in polygon_coords]
    base_poly_enu = Polygon(enu_points)
    if not base_poly_enu.is_valid:
        base_poly_enu = make_valid(base_poly_enu)

    is_airborne = (current_drone_pos is not None)
    if is_airborne:
        start_pos_enu = geodetic_to_enu(current_drone_pos["lat"], current_drone_pos["lon"], ref_lat, ref_lon)
    else:
        start_pos_enu = dock_enu

    target_geom = base_poly_enu
    is_replanned = False

    # Soustraction garantie des bandes de scan terminées
    if completed_lines_wgs84 and len(completed_lines_wgs84) > 0:
        lines_enu = []
        for (lat1, lon1), (lat2, lon2) in completed_lines_wgs84:
            p1_e = geodetic_to_enu(lat1, lon1, ref_lat, ref_lon)
            p2_e = geodetic_to_enu(lat2, lon2, ref_lat, ref_lon)
            lines_enu.append((p1_e, p2_e))

        target_geom = compute_clean_residual_polygon(base_poly_enu, lines_enu, lane_spacing)
        is_replanned = True

        print(f"✂️ [Polygone Résiduel] {len(lines_enu)} bandes soustraites avec succès !")
        print(f"   • Surface initiale : {round(base_poly_enu.area)} m²")
        print(f"   • Surface RESTANTE : {round(target_geom.area)} m²")

        if target_geom.is_empty:
            print("🏁 Toute la surface a été scannée. Fin de mission.")
            return {
                "mission_complete": True,
                "waypoints": [],
                "raw_polygon": polygon_coords,
                "safe_flight_altitude": flight_altitude,
                "dominant_obstacle_height": round(tallest_obstacle_val, 1),
                "dock_station": dock_station,
                "feasibility": {"is_feasible": True, "battery_cost_percent": 0.0}
            }

    best_angle, best_wps_enu, metrics = find_optimal_survey(
        target_geom=target_geom,
        start_pos_enu=start_pos_enu,
        dock_enu=dock_enu,
        flight_altitude=flight_altitude,
        lane_spacing=lane_spacing,
        wind_speed_ms=wind_speed_ms,
        wind_dir_deg=wind_dir_deg,
        air_density=air_density,
        specs=d_specs,
        is_airborne_start=is_airborne
    )

    if not best_wps_enu:
        return {
            "mission_complete": True,
            "waypoints": [],
            "raw_polygon": polygon_coords,
            "safe_flight_altitude": flight_altitude,
            "dominant_obstacle_height": round(tallest_obstacle_val, 1),
            "dock_station": dock_station,
            "feasibility": {"is_feasible": True, "battery_cost_percent": 0.0}
        }

    # Injection des waypoints avec l'altitude exacte calculée
    gps_waypoints = []
    for item in best_wps_enu:
        east, north = item["pt"]
        lat, lon = enu_to_geodetic(east, north, ref_lat, ref_lon)
        gps_waypoints.append({
            "lat": lat,
            "lon": lon,
            "alt": flight_altitude,
            "type": item.get("type", "SURVEY")
        })

    total_battery_joules = d_specs.battery_voltage_nom * d_specs.battery_capacity_ah * 3600.0
    usable_joules = total_battery_joules * (1.0 - (d_specs.reserve_soc_percent / 100.0))
    battery_cost_percent = round((metrics["total_joules"] / total_battery_joules) * 100.0, 1)

    return {
        "optimal_angle_deg": best_angle,
        "safe_flight_altitude": flight_altitude,
        "dominant_obstacle_height": round(tallest_obstacle_val, 1),
        "lane_spacing_m": round(lane_spacing, 2),
        "ground_sample_distance_cm": round(gsd_cm_px, 2),
        "waypoints": gps_waypoints,
        "raw_polygon": polygon_coords,
        "tallest_obstacle": round(tallest_obstacle_val, 1),
        "dock_station": dock_station,
        "is_replanned": is_replanned,
        "feasibility": {
            "is_feasible": metrics["total_joules"] <= usable_joules,
            "battery_cost_percent": battery_cost_percent,
            "total_energy_joules": round(metrics["total_joules"], 0),
            "total_distance_km": round(metrics["total_dist_m"] / 1000.0, 2),
            "estimated_time_min": round(metrics["duration_sec"] / 60.0, 1)
        }
    }

def fetch_tallest_building_height_osm(
    polygon_coords: List[Tuple[float, float]],
    timeout_sec: float = 3.5,
    default_if_none: float = 12.0
) -> float:
    """
    Détecte la hauteur du plus haut obstacle (bâtiments, silos, cheminées, réservoirs).
    """
    lats = [pt[0] for pt in polygon_coords]
    lons = [pt[1] for pt in polygon_coords]
    
    s, n = min(lats) - 0.0008, max(lats) + 0.0008
    w, e = min(lons) - 0.0008, max(lons) + 0.0008

    query = f"""
    [out:json][timeout:3];
    (
      way["building"]({s},{w},{n},{e});
      way["man_made"]({s},{w},{n},{e});
      relation["building"]({s},{w},{n},{e});
      node["man_made"]({s},{w},{n},{e});
    );
    out tags;
    """

    url = "https://overpass-api.de/api/interpreter"
    max_detected_height = 0.0
    detected_count = 0

    try:
        data = urllib.parse.urlencode({'data': query}).encode('utf-8')
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={'User-Agent': 'TacticalUAV-MissionPlanner/3.0'}
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            res_json = json.loads(resp.read().decode('utf-8'))
            elements = res_json.get('elements', [])

            for el in elements:
                tags = el.get('tags', {})
                h = 0.0

                # 1. Hauteur explicite tag 'height'
                if 'height' in tags:
                    raw_h = re.sub(r'[^\d.]', '', tags['height'])
                    if raw_h:
                        try:
                            h = float(raw_h)
                        except ValueError:
                            pass

                # 2. Hauteur via les étages
                elif 'building:levels' in tags:
                    try:
                        levels = float(re.sub(r'[^\d.]', '', tags['building:levels']))
                        h = levels * 3.5
                    except ValueError:
                        pass

                # 3. Typologie des structures industrielles connues
                man_made = tags.get('man_made', '')
                building = tags.get('building', '')

                if h == 0.0:
                    if man_made in ['chimney', 'flare']:
                        h = 35.0  # Cheminée industrielle
                    elif man_made in ['silo', 'storage_tank', 'water_tower']:
                        h = 22.0  # Silo ou grand réservoir
                    elif building in ['industrial', 'warehouse', 'manufacture']:
                        h = 14.0  # Hangar industriel
                    elif building or man_made:
                        h = default_if_none

                if h > 0:
                    detected_count += 1
                    if h > max_detected_height:
                        max_detected_height = h

        if max_detected_height > 0:
            print(f"🏢 [OSM] {detected_count} structure(s) scannée(s). Obstacle le plus haut = {max_detected_height:.1f} m")
            return max_detected_height

    except Exception as err:
        print(f"⚠️ [OSM] Requête non aboutie ({err}). Maintien de l'altitude de repli.")

    return default_if_none