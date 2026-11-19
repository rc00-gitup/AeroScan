"""
Industrial UAV Dynamic Flight Simulator & Autopilot
==================================================
- Simulateur cinématique asservi temps réel (dt = 0.05s).
- Intégration photogrammétrique active : déclenchement automatique des clichés.
- Double sécurité d'enregistrement des lignes de scan (zéro répétition).
- Enregistrement serveur de l'historique réel de vol (indépendant du navigateur).
- Arrêt stationnaire pur lors de l'adaptation avec affichage du plan avant transit.
- Gestion des 3 modes de navigation (one_way, round_trip, infinite).
- Failsafes certifiés : RTL d'urgence (Vent > 10.5 m/s, Température > 40°C, Batterie <= 30%).
- Arrêt d'urgence avec retour effectif à la borne (CFIT-Free).
"""

import math
import threading
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

import planner
from planner import DroneSpecs, geodetic_to_enu, enu_to_geodetic, compute_multirotor_power
from battery_model import LiPo6SBattery
from weather_engine import WeatherEngine
from mapping_engine import engine as mapping_engine

# ==============================================================================
# CONFIGURATION MATÉRIELLE & REGISTRE SYSTÈME
# ==============================================================================

SPECS = DroneSpecs()
weather_env = WeatherEngine()
battery = LiPo6SBattery(capacity_mah=SPECS.battery_capacity_ah * 1000.0)

_STATE_LOCK = threading.Lock()
_REPLAN_EVENT = threading.Event()
_ABORT_EVENT = threading.Event()

_CURRENT_MISSION_ID = 0
_ACTIVE_MISSION: Optional[Dict[str, Any]] = None
_NAV_MODE: str = "one_way"

# Gestion du déclenchement caméra (Photogrammétrie)
_PHOTO_TRIGGER_DIST: float = 12.0
_DIST_SINCE_LAST_PHOTO: float = 0.0

# Historique de vol stocké sur le serveur (1 point tous les 2 mètres)
_FLOWN_HISTORY: List[List[float]] = []
_LAST_LOG_E: float = 0.0
_LAST_LOG_N: float = 0.0

LIVE_DRONE_STATE: Dict[str, Any] = {
    "lat": 32.2515,
    "lon": -9.2630,
    "alt": 0.0,
    "speed": 0.0,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "battery": 100,
    "voltage": 25.2,
    "current_amps": 0.0,
    "power_watts": 0.0,
    "cell_temp_c": 25.0,
    "status": "DOCKING STATION READY",
    "cross_track_error": 0.0,
    "dock_station": {"lat": 32.2515, "lng": -9.2630},
    "wind_speed": 0.0,
    "wind_dir": 0.0,
    "temp_c": 20.0,
    "weather_label": "Ready",
    "mission_progress_percent": 0.0,
    "plan_version": 1,
    "remaining_waypoints": [],
    "nav_mode": "one_way",
    "flown_history": []
}


# ==============================================================================
# INTERFACES DE CONTRÔLE GCS (THREAD-SAFE)
# ==============================================================================

def get_telemetry() -> Dict[str, Any]:
    with _STATE_LOCK:
        return dict(LIVE_DRONE_STATE)

def get_active_mission() -> Optional[Dict[str, Any]]:
    with _STATE_LOCK:
        if _ACTIVE_MISSION is None:
            return None
        return dict(_ACTIVE_MISSION)

def set_weather_profile(mode: str) -> Dict[str, Any]:
    with _STATE_LOCK:
        dock = LIVE_DRONE_STATE.get("dock_station", {"lat": 32.2515, "lng": -9.2630})
        dock_lat = float(dock["lat"])
        dock_lon = float(dock.get("lon", dock.get("lng", -9.2630)))

    summary = weather_env.set_profile(mode, dock_lat, dock_lon)

    with _STATE_LOCK:
        if _ACTIVE_MISSION is not None and "raw_polygon" in _ACTIVE_MISSION:
            _REPLAN_EVENT.set()

    return summary

def set_nav_mode(mode: str):
    """Bascule le mode de vol : 'one_way', 'round_trip', 'infinite'."""
    global _NAV_MODE
    if mode in ["one_way", "round_trip", "infinite"]:
        with _STATE_LOCK:
            _NAV_MODE = mode
            LIVE_DRONE_STATE["nav_mode"] = mode
        print(f"🧭 [Autopilot] Mode de vol sélectionné : {mode.upper()}")

def abort_mission():
    """Déclenche immédiatement l'interruption d'urgence et le retour effectif au Dock."""
    print("🛑 [Autopilot] Arrêt d'urgence reçu. Déclenchement du RTL vers la borne.")
    _ABORT_EVENT.set()


# ==============================================================================
# CINÉMATIQUE ASSERVIE & MOTEUR PHYSIQUE AVEC CAPTURE PHOTO
# ==============================================================================

def _update_telemetry(status: str, b_state: Dict[str, Any], w_sample: Dict[str, Any], **kwargs):
    with _STATE_LOCK:
        LIVE_DRONE_STATE["status"] = status
        LIVE_DRONE_STATE["battery"] = b_state["battery_percent"]
        LIVE_DRONE_STATE["voltage"] = b_state["voltage"]
        LIVE_DRONE_STATE["current_amps"] = b_state["current_amps"]
        LIVE_DRONE_STATE["power_watts"] = b_state["power_watts"]
        LIVE_DRONE_STATE["cell_temp_c"] = b_state["cell_temperature_c"]
        LIVE_DRONE_STATE["wind_speed"] = w_sample["wind_speed_ms"]
        LIVE_DRONE_STATE["wind_dir"] = w_sample["wind_direction_deg"]
        LIVE_DRONE_STATE["temp_c"] = w_sample["temperature_c"]
        LIVE_DRONE_STATE["weather_label"] = weather_env.label

        for key, value in kwargs.items():
            LIVE_DRONE_STATE[key] = value

def _fly_segment_enu(
    p_start: List[float],
    p_end: List[float],
    target_speed: float,
    status_label: str,
    mission_id: int,
    ref_lat: float,
    ref_lon: float,
    dt: float = 0.05,
    is_survey: bool = False
) -> bool:
    global _FLOWN_HISTORY, _LAST_LOG_E, _LAST_LOG_N, _DIST_SINCE_LAST_PHOTO, _PHOTO_TRIGGER_DIST

    dx = p_end[0] - p_start[0]
    dy = p_end[1] - p_start[1]
    segment_dist = math.hypot(dx, dy)
    if segment_dist < 0.1:
        return True

    track_bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
    elapsed_dist = 0.0

    while elapsed_dist < segment_dist:
        if mission_id != _CURRENT_MISSION_ID or _ABORT_EVENT.is_set():
            return False

        step_dist = target_speed * dt
        elapsed_dist = min(segment_dist, elapsed_dist + step_dist)
        alpha = elapsed_dist / segment_dist

        cur_e = p_start[0] + alpha * dx
        cur_n = p_start[1] + alpha * dy

        w_sample = weather_env.sample_instantaneous_wind()
        w_speed = w_sample["wind_speed_ms"]
        w_dir = w_sample["wind_direction_deg"]
        air_density = w_sample["air_density_kg_m3"]

        w_blow_rad = math.radians((w_dir + 180.0) % 360.0)
        track_rad = math.radians(track_bearing)
        wind_rel_angle = w_blow_rad - track_rad
        cross_wind = w_speed * math.sin(wind_rel_angle)
        head_wind = -w_speed * math.cos(wind_rel_angle)

        effective_ground_speed = max(1.0, target_speed - (head_wind * 0.1))
        crab_angle = math.degrees(math.atan2(cross_wind, effective_ground_speed))
        commanded_yaw = (track_bearing + crab_angle) % 360.0

        cross_track_err = round(0.08 * cross_wind * w_sample["turbulence_severity"], 2)
        drift_e = -cross_track_err * math.cos(track_rad)
        drift_n = cross_track_err * math.sin(track_rad)

        cur_lat, cur_lon = enu_to_geodetic(cur_e + drift_e, cur_n + drift_n, ref_lat, ref_lon)

        # Enregistrement de la trace de vol (tous les 2 mètres)
        if not _FLOWN_HISTORY or math.hypot(cur_e - _LAST_LOG_E, cur_n - _LAST_LOG_N) >= 2.0:
            _FLOWN_HISTORY.append([cur_lat, cur_lon])
            _LAST_LOG_E, _LAST_LOG_N = cur_e, cur_n
            with _STATE_LOCK:
                LIVE_DRONE_STATE["flown_history"] = list(_FLOWN_HISTORY)

        # ======================================================================
        # DÉCLENCHEMENT DE PRISE DE VUE PHOTOGRAMMÉTRIQUE ASSERVIE
        # ======================================================================
        if is_survey:
            _DIST_SINCE_LAST_PHOTO += step_dist
            if _DIST_SINCE_LAST_PHOTO >= _PHOTO_TRIGGER_DIST:
                _DIST_SINCE_LAST_PHOTO = 0.0
                with _STATE_LOCK:
                    drone_alt = LIVE_DRONE_STATE.get("alt", 20.0)
                # Prise de photo HD réelle
                mapping_engine.trigger_capture(cur_lat, cur_lon, drone_alt, commanded_yaw)

        roll_angle = round(max(-30.0, min(30.0, crab_angle * 0.8)), 1)
        pitch_angle = round(max(-35.0, min(15.0, -4.0 - (target_speed * 1.5) - (head_wind * 0.8))), 1)

        power = compute_multirotor_power(target_speed, track_bearing, w_speed, w_dir, air_density, SPECS)
        b_state = battery.update(power, w_sample["temperature_c"], dt=dt)

        _update_telemetry(
            status=status_label,
            b_state=b_state,
            w_sample=w_sample,
            lat=cur_lat,
            lon=cur_lon,
            speed=round(effective_ground_speed, 1),
            yaw=round(commanded_yaw, 1),
            pitch=pitch_angle,
            roll=roll_angle,
            cross_track_error=abs(cross_track_err)
        )

        if b_state["is_critical"]:
            return False

        time.sleep(dt)

    return True


# ==============================================================================
# SÉCURITÉ : PROCÉDURE RTL & RECHARGE AU DOCK
# ==============================================================================

def _execute_safe_rtl(dock_lat: float, dock_lon: float, obstacle_ceiling: float, ref_lat: float, ref_lon: float, mission_id: int):
    dt = 0.05
    with _STATE_LOCK:
        c_lat = LIVE_DRONE_STATE["lat"]
        c_lon = LIVE_DRONE_STATE["lon"]
        c_alt = LIVE_DRONE_STATE["alt"]

    _ABORT_EVENT.clear()

    c_e, c_n = geodetic_to_enu(c_lat, c_lon, ref_lat, ref_lon)
    d_e, d_n = geodetic_to_enu(dock_lat, dock_lon, ref_lat, ref_lon)

    # 1. Montée de sécurité
    safe_rtl_alt = max(c_alt, obstacle_ceiling + 10.0)
    while c_alt < safe_rtl_alt:
        c_alt = min(safe_rtl_alt, c_alt + (SPECS.climb_speed_ms * dt))
        w_sample = weather_env.sample_instantaneous_wind()
        power = compute_multirotor_power(0.0, 0.0, w_sample["wind_speed_ms"], w_sample["wind_direction_deg"], 1.225, SPECS) + 200.0
        b_state = battery.update(power, w_sample["temperature_c"], dt=dt)
        _update_telemetry("RTL: CLIMB TO SAFE ALTITUDE", b_state, w_sample, alt=round(c_alt, 1), speed=0.0)
        time.sleep(dt)

    # 2. Transit vers la borne (pas de survey)
    _fly_segment_enu([c_e, c_n], [d_e, d_n], SPECS.transit_speed_ms, "RTL: RETURNING TO DOCK", mission_id, ref_lat, ref_lon, dt, is_survey=False)

    # 3. Descente contrôlée uniquement au-dessus du dock
    while c_alt > 0.0:
        c_alt = max(0.0, c_alt - (SPECS.descent_speed_ms * dt))
        w_sample = weather_env.sample_instantaneous_wind()
        power = compute_multirotor_power(0.0, 0.0, w_sample["wind_speed_ms"], w_sample["wind_direction_deg"], 1.225, SPECS) * 0.75
        b_state = battery.update(power, w_sample["temperature_c"], dt=dt)
        _update_telemetry("RTL: LANDING ON DOCK", b_state, w_sample, alt=round(c_alt, 1), speed=0.0, pitch=0.0, roll=0.0)
        time.sleep(dt)

    with _STATE_LOCK:
        LIVE_DRONE_STATE["status"] = "DOCK: CHARGING"
        LIVE_DRONE_STATE["speed"] = 0.0
        LIVE_DRONE_STATE["alt"] = 0.0

def _dock_recharge_and_finish(mission_id: Optional[int] = None):
    dt = 0.08
    print("🔌 [Dock] Drone sécurisé sur la borne. Début de la recharge active...")

    while battery.soc < 1.0:
        # Si une nouvelle mission démarre, on stoppe immédiatement ce thread fantôme
        if mission_id is not None and mission_id != _CURRENT_MISSION_ID:
            print("🛑 [Dock] Nouvelle mission détectée : arrêt immédiat de la boucle de charge précédente.")
            return

        with _STATE_LOCK:
            battery.remaining_capacity_mah = min(battery.nominal_capacity_mah, battery.remaining_capacity_mah + 220.0)
            battery.soc = battery.remaining_capacity_mah / battery.nominal_capacity_mah
            battery.cell_temperature_c = max(22.0, battery.cell_temperature_c - 0.5)
            pct = int(battery.soc * 100)

        w_sample = weather_env.sample_instantaneous_wind()
        b_state = {
            "battery_percent": pct,
            "voltage": round(21.2 + 4.0 * battery.soc, 1),
            "current_amps": -18.5,
            "power_watts": -420.0,
            "cell_temperature_c": battery.cell_temperature_c
        }

        _update_telemetry(f"DOCK: CHARGING ({pct}%)", b_state, w_sample, alt=0.0, speed=0.0, pitch=0.0, roll=0.0)
        time.sleep(dt)

    battery.reset()
    print("🛑 [Dock] Batterie à 100%. Mission clôturée à la borne.")

    with _STATE_LOCK:
        if mission_id is None or mission_id == _CURRENT_MISSION_ID:
            LIVE_DRONE_STATE["status"] = "RTL FINISHED: DOCKED"
            LIVE_DRONE_STATE["speed"] = 0.0
            LIVE_DRONE_STATE["alt"] = 0.0
            LIVE_DRONE_STATE["remaining_waypoints"] = []

# ==============================================================================
# BOUCLE PRINCIPALE DE VOL (AUTOPILOT THREAD)
# ==============================================================================

def _flight_loop(mission: Dict[str, Any], dock_station: Dict[str, Any], mission_id: int):
    global _ACTIVE_MISSION, _PHOTO_TRIGGER_DIST, _DIST_SINCE_LAST_PHOTO
    dt = 0.05

    try:
        waypoints = mission.get("waypoints", [])
        if not waypoints:
            return

        target_alt = mission.get("safe_flight_altitude", 25.0)
        obstacle_height = mission.get("dominant_obstacle_height", target_alt - 10.0)
        dock_lat = float(dock_station["lat"])
        dock_lon = float(dock_station.get("lon", dock_station.get("lng", 0.0)))
        ref_lat, ref_lon = waypoints[0]["lat"], waypoints[0]["lon"]
        lane_spacing = mission.get("lane_spacing_m", 15.0)

        _PHOTO_TRIGGER_DIST = mapping_engine.compute_trigger_distance(target_alt)
        _DIST_SINCE_LAST_PHOTO = _PHOTO_TRIGGER_DIST

        d_e, d_n = geodetic_to_enu(dock_lat, dock_lon, ref_lat, ref_lon)

        with _STATE_LOCK:
            LIVE_DRONE_STATE["lat"] = dock_lat
            LIVE_DRONE_STATE["lon"] = dock_lon
            LIVE_DRONE_STATE["alt"] = 0.0
            LIVE_DRONE_STATE["status"] = "TAKEOFF"
            LIVE_DRONE_STATE["remaining_waypoints"] = list(waypoints)

        cur_alt = 0.0
        while cur_alt < target_alt:
            if mission_id != _CURRENT_MISSION_ID or _ABORT_EVENT.is_set():
                return
            cur_alt = min(target_alt, cur_alt + (SPECS.climb_speed_ms * dt))
            w_sample = weather_env.sample_instantaneous_wind()
            power = compute_multirotor_power(0.0, 0.0, w_sample["wind_speed_ms"], w_sample["wind_direction_deg"], 1.225, SPECS) + 150.0
            b_state = battery.update(power, w_sample["temperature_c"], dt=dt)
            _update_telemetry("TAKEOFF", b_state, w_sample, alt=round(cur_alt, 1))
            time.sleep(dt)

        wp0_e, wp0_n = geodetic_to_enu(waypoints[0]["lat"], waypoints[0]["lon"], ref_lat, ref_lon)
        success = _fly_segment_enu([d_e, d_n], [wp0_e, wp0_n], SPECS.transit_speed_ms, "TRANSIT OUTBOUND", mission_id, ref_lat, ref_lon, dt, is_survey=False)
        if not success:
            _execute_safe_rtl(dock_lat, dock_lon, obstacle_height, ref_lat, ref_lon, mission_id)
            _dock_recharge_and_finish(mission_id)
            return

        current_wps = list(waypoints)
        current_idx = 0
        round_trip_done = False
        completed_scan_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

        w_initial = weather_env.get_summary()
        last_planned_wind_dir = float(mission.get("wind_dir_deg", w_initial.get("wind_direction_deg", 0.0)))
        last_replan_timestamp = time.time()

        while True:
            while current_idx < len(current_wps) - 1:
                if mission_id != _CURRENT_MISSION_ID:
                    return

                if _ABORT_EVENT.is_set():
                    _execute_safe_rtl(dock_lat, dock_lon, obstacle_height, ref_lat, ref_lon, mission_id)
                    _dock_recharge_and_finish(mission_id)
                    return

                w_sample = weather_env.sample_instantaneous_wind()
                if w_sample["wind_speed_ms"] > 10.5 or w_sample["temperature_c"] > 40.0:
                    _execute_safe_rtl(dock_lat, dock_lon, obstacle_height, ref_lat, ref_lon, mission_id)
                    _dock_recharge_and_finish(mission_id)
                    return

                if _REPLAN_EVENT.is_set():
                    _REPLAN_EVENT.clear()
                    now = time.time()
                    w_sum = weather_env.get_summary()
                    new_w_speed = w_sum["wind_speed_ms"]
                    new_w_dir = w_sum["wind_direction_deg"]

                    if new_w_speed < 3.0:
                        continue

                    if (now - last_replan_timestamp) < 30.0:
                        continue

                    if (len(current_wps) - current_idx) < 3:
                        continue

                    delta_wind = abs((new_w_dir - last_planned_wind_dir + 90.0) % 180.0 - 90.0)

                    if delta_wind < 25.0:
                        continue

                    with _STATE_LOCK:
                        cur_lat = LIVE_DRONE_STATE["lat"]
                        cur_lon = LIVE_DRONE_STATE["lon"]
                        cur_status = LIVE_DRONE_STATE.get("status", "")
                        mission_ref = dict(_ACTIVE_MISSION)

                    new_density = weather_env.get_air_density(altitude_m=target_alt)

                    lines_to_subtract = list(completed_scan_lines)
                    if "SURVEY" in cur_status and current_idx < len(current_wps):
                        last_wp = current_wps[current_idx]
                        lines_to_subtract.append((
                            (last_wp["lat"], last_wp["lon"]),
                            (cur_lat, cur_lon)
                        ))

                    candidate_plan = planner.plan_mission(
                        polygon_coords=mission_ref["raw_polygon"],
                        dock_station=dock_station,
                        wind_speed_ms=new_w_speed,
                        wind_dir_deg=new_w_dir,
                        air_density=new_density,
                        current_drone_pos={"lat": cur_lat, "lon": cur_lon},
                        completed_lines_wgs84=lines_to_subtract
                    )

                    if candidate_plan.get("mission_complete", False) or not candidate_plan.get("waypoints"):
                        break

                    with _STATE_LOCK:
                        freeze_lat = LIVE_DRONE_STATE["lat"]
                        freeze_lon = LIVE_DRONE_STATE["lon"]
                        LIVE_DRONE_STATE["speed"] = 0.0
                        LIVE_DRONE_STATE["pitch"] = 0.0
                        LIVE_DRONE_STATE["roll"] = 0.0

                    new_wps = candidate_plan["waypoints"]
                    completed_scan_lines = lines_to_subtract
                    last_planned_wind_dir = new_w_dir
                    last_replan_timestamp = now
                    _DIST_SINCE_LAST_PHOTO = _PHOTO_TRIGGER_DIST

                    with _STATE_LOCK:
                        _ACTIVE_MISSION = candidate_plan
                        LIVE_DRONE_STATE["remaining_waypoints"] = list(new_wps)
                        LIVE_DRONE_STATE["plan_version"] += 1
                        LIVE_DRONE_STATE["status"] = "HOVER: REORIENTING TO WIND"

                    for _ in range(15):
                        w_sample = weather_env.sample_instantaneous_wind()
                        p_hov = compute_multirotor_power(0.0, 0.0, w_sample["wind_speed_ms"], w_sample["wind_direction_deg"], 1.225, SPECS)
                        b_state = battery.update(p_hov, w_sample["temperature_c"], dt=dt)
                        _update_telemetry("HOVER: NEW WIND ADAPTATION", b_state, w_sample, lat=freeze_lat, lon=freeze_lon, speed=0.0, pitch=0.0, roll=0.0)
                        time.sleep(dt)

                    next_e, next_n = geodetic_to_enu(new_wps[0]["lat"], new_wps[0]["lon"], ref_lat, ref_lon)
                    c_e, c_n = geodetic_to_enu(freeze_lat, freeze_lon, ref_lat, ref_lon)
                    _fly_segment_enu([c_e, c_n], [next_e, next_n], SPECS.transit_speed_ms, "TRANSIT TO OPTIMAL SECTOR", mission_id, ref_lat, ref_lon, dt, is_survey=False)

                    current_wps = new_wps
                    current_idx = 0
                    lane_spacing = candidate_plan["lane_spacing_m"]
                    continue

                p1 = current_wps[current_idx]
                p2 = current_wps[current_idx + 1]
                e1, n1 = geodetic_to_enu(p1["lat"], p1["lon"], ref_lat, ref_lon)
                e2, n2 = geodetic_to_enu(p2["lat"], p2["lon"], ref_lat, ref_lon)

                seg_len = math.hypot(e2 - e1, n2 - n1)
                is_scan = (p1.get("type") == "SURVEY" and p2.get("type") == "SURVEY") or (seg_len > (lane_spacing * 1.3) and "SURVEY" in LIVE_DRONE_STATE["status"])

                speed = SPECS.cruise_speed_ms if is_scan else SPECS.turn_speed_ms
                status_text = "SURVEY ACTIVE" if is_scan else ("LINE TURN" if p1.get("type") == "TURN" else "TRANSIT")

                ok = _fly_segment_enu([e1, n1], [e2, n2], speed, status_text, mission_id, ref_lat, ref_lon, dt, is_survey=is_scan)
                if not ok:
                    _execute_safe_rtl(dock_lat, dock_lon, obstacle_height, ref_lat, ref_lon, mission_id)
                    _dock_recharge_and_finish(mission_id)
                    return

                if is_scan:
                    completed_scan_lines.append(((p1["lat"], p1["lon"]), (p2["lat"], p2["lon"])))
                    _DIST_SINCE_LAST_PHOTO = _PHOTO_TRIGGER_DIST

                current_idx += 1
                with _STATE_LOCK:
                    progress = round((current_idx / (len(current_wps) - 1)) * 100.0, 1)
                    LIVE_DRONE_STATE["mission_progress_percent"] = progress
                    LIVE_DRONE_STATE["remaining_waypoints"] = current_wps[current_idx:]

            if _NAV_MODE == "one_way":
                break
            elif _NAV_MODE == "round_trip":
                if not round_trip_done:
                    round_trip_done = True
                    current_wps = list(reversed(current_wps))
                    current_idx = 0
                    with _STATE_LOCK:
                        LIVE_DRONE_STATE["remaining_waypoints"] = list(current_wps)
                        LIVE_DRONE_STATE["plan_version"] += 1
                    continue
                else:
                    break
            elif _NAV_MODE == "infinite":
                current_wps = list(reversed(current_wps))
                current_idx = 0
                with _STATE_LOCK:
                    LIVE_DRONE_STATE["remaining_waypoints"] = list(current_wps)
                    LIVE_DRONE_STATE["plan_version"] += 1
                continue

        last_wp = current_wps[-1]
        l_e, l_n = geodetic_to_enu(last_wp["lat"], last_wp["lon"], ref_lat, ref_lon)
        _fly_segment_enu([l_e, l_n], [d_e, d_n], SPECS.transit_speed_ms, "TRANSIT HOME TO DOCK", mission_id, ref_lat, ref_lon, dt, is_survey=False)

        cur_alt = target_alt
        while cur_alt > 0.0:
            if mission_id != _CURRENT_MISSION_ID:
                return
            cur_alt = max(0.0, cur_alt - (SPECS.descent_speed_ms * dt))
            w_sample = weather_env.sample_instantaneous_wind()
            power = compute_multirotor_power(0.0, 0.0, w_sample["wind_speed_ms"], w_sample["wind_direction_deg"], 1.225, SPECS) * 0.75
            b_state = battery.update(power, w_sample["temperature_c"], dt=dt)
            _update_telemetry("LANDING ON DOCK", b_state, w_sample, alt=round(cur_alt, 1), speed=0.0)
            time.sleep(dt)

        _dock_recharge_and_finish(mission_id)

    except Exception as e:
        pass

def start_mission(mission_plan: Dict[str, Any], dock_station: Dict[str, Any], mode: str = "one_way"):
    global _ACTIVE_MISSION, _CURRENT_MISSION_ID, _NAV_MODE, _FLOWN_HISTORY, _LAST_LOG_E, _LAST_LOG_N, _DIST_SINCE_LAST_PHOTO

    with _STATE_LOCK:
        _CURRENT_MISSION_ID += 1
        mission_id = _CURRENT_MISSION_ID
        _ABORT_EVENT.clear()
        _REPLAN_EVENT.clear()          # <-- VITAL : purge les replanifications antérieures
        battery.reset()                # <-- VITAL : réinitialise la batterie à 100% pour la nouvelle mission
        _ACTIVE_MISSION = mission_plan
        _NAV_MODE = mode
        LIVE_DRONE_STATE["nav_mode"] = mode
        LIVE_DRONE_STATE["status"] = "TAKEOFF"
        LIVE_DRONE_STATE["alt"] = 0.0
        LIVE_DRONE_STATE["speed"] = 0.0
        _FLOWN_HISTORY = []
        _LAST_LOG_E, _LAST_LOG_N = 0.0, 0.0
        LIVE_DRONE_STATE["flown_history"] = []
        _DIST_SINCE_LAST_PHOTO = 999.0

    threading.Thread(
        target=_flight_loop,
        args=(mission_plan, dock_station, mission_id),
        daemon=True,
        name=f"UAV-FlightLoop-ID{mission_id}"
    ).start()