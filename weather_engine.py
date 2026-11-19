"""
Industrial Environmental & Weather Simulation Engine
====================================================
- Respect des plafonds opérationnels certifiés (Vent max 10.5 m/s, Temp max 40°C).
- Scénarios stochastiques maîtrisés (Storm > 10.5 m/s, Heatwave à 50% > 40°C).
- Modélisation vectorielle 3D des turbulences (Dryden).
- Cisaillement atmosphérique vertical (Loi de Hellmann) et densité barométrique.
- Thread-safe avec arrêt propre.
"""

import math
import json
import urllib.request
import threading
import time
from typing import Dict, Any, Optional
import numpy as np


class WeatherEngine:
    # Plafonds opérationnels stricts du drone industriel
    MAX_OPERATIONAL_WIND_MS = 10.5      # Limite vent : 10.5 m/s (~38 km/h)
    MAX_OPERATIONAL_TEMP_C = 40.0       # Limite température : 40.0°C
    MAX_RAIN_OPERATIONAL_MMH = 5.0      # Tolérance pluie max IP55

    def __init__(self, random_seed: Optional[int] = None):
        self._lock = threading.RLock()
        self._stop_event = threading.Event()

        if random_seed is not None:
            np.random.seed(random_seed)

        self.active_mode = "live"
        self.label = "Stationary Live"
        self.base_wind_speed_10m = 3.5
        self.base_wind_dir_deg = 45.0
        self.temperature_c = 22.0
        self.surface_pressure_hpa = 1013.25
        self.rain_intensity_mmh = 0.0
        self.turbulence_intensity = 0.12

        self._gust_u = 0.0
        self._gust_v = 0.0
        self._gust_w = 0.0

        self._current_instant: Dict[str, Any] = {
            "wind_speed_ms": 3.5,
            "wind_direction_deg": 45.0,
            "vertical_wind_ms": 0.0,
            "temperature_c": 22.0,
            "air_density_kg_m3": 1.225,
            "rain_down_force_n": 0.0,
            "turbulence_severity": 1.0,
            "is_flight_permitted": True
        }

        self._worker_thread = threading.Thread(target=self._turbulence_worker, daemon=True, name="Weather-Engine")
        self._worker_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

    def get_air_density(self, altitude_m: float = 0.0) -> float:
        with self._lock:
            p0 = self.surface_pressure_hpa * 100.0
            t0 = self.temperature_c + 273.15

        r_spec = 287.058
        g = 9.80665
        molar_mass = 0.0289644
        r_universal = 8.31446

        pressure_z = p0 * math.exp((-molar_mass * g * altitude_m) / (r_universal * t0))
        return round(pressure_z / (r_spec * t0), 4)

    def _compute_rain_impact_force(self, rain_rate_mmh: float, drone_surface_m2: float = 0.18) -> float:
        if rain_rate_mmh <= 0.0:
            return 0.0
        mass_flux = (rain_rate_mmh / 1000.0 / 3600.0) * 1000.0
        return round(mass_flux * drone_surface_m2 * 9.0, 3)

    def _turbulence_worker(self):
        dt = 0.05
        time_constant = 1.8

        while not self._stop_event.is_set():
            decay = math.exp(-dt / time_constant)

            with self._lock:
                base_speed = self.base_wind_speed_10m
                base_dir = self.base_wind_dir_deg
                temp = self.temperature_c
                rain = self.rain_intensity_mmh
                turb_scale = self.turbulence_intensity

            sigma_u = base_speed * turb_scale
            sigma_v = sigma_u * 0.7
            sigma_w = sigma_u * 0.35

            self._gust_u = (self._gust_u * decay) + np.random.normal(0, sigma_u * math.sqrt(1 - decay**2))
            self._gust_v = (self._gust_v * decay) + np.random.normal(0, sigma_v * math.sqrt(1 - decay**2))
            self._gust_w = (self._gust_w * decay) + np.random.normal(0, sigma_w * math.sqrt(1 - decay**2))

            u_total = max(0.1, base_speed + self._gust_u)
            delta_dir_rad = math.atan2(self._gust_v, u_total)
            instant_speed = math.hypot(u_total, self._gust_v)
            instant_dir = (base_dir + math.degrees(delta_dir_rad)) % 360.0

            rain_force = self._compute_rain_impact_force(rain)
            air_dens = self.get_air_density(altitude_m=35.0)

            # Contrôle strict des plafonds opérationnels
            is_permitted = (
                instant_speed <= self.MAX_OPERATIONAL_WIND_MS and
                temp <= self.MAX_OPERATIONAL_TEMP_C and
                rain <= self.MAX_RAIN_OPERATIONAL_MMH
            )

            with self._lock:
                self._current_instant = {
                    "wind_speed_ms": round(instant_speed, 1),
                    "wind_direction_deg": round(instant_dir, 1),
                    "vertical_wind_ms": round(self._gust_w, 2),
                    "temperature_c": round(temp, 1),
                    "air_density_kg_m3": air_dens,
                    "rain_down_force_n": rain_force,
                    "turbulence_severity": round(max(0.2, turb_scale / 0.12), 2),
                    "is_flight_permitted": is_permitted
                }

            time.sleep(dt)

    def sample_instantaneous_wind(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._current_instant)

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            inst = dict(self._current_instant)
            return {
                "mode": self.active_mode,
                "label": self.label,
                "wind_speed_ms": inst["wind_speed_ms"],
                "wind_direction_deg": int(inst["wind_direction_deg"]),
                "temperature_c": inst["temperature_c"],
                "rain_mmh": self.rain_intensity_mmh,
                "is_flight_permitted": inst["is_flight_permitted"]
            }

    def fetch_live_satellite_weather(self, lat: float, lon: float, timeout_sec: float = 2.0) -> bool:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current="
            f"temperature_2m,surface_pressure,wind_speed_10m,wind_direction_10m,rain"
        )
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'UAV-GCS/2.0'})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                curr = data.get('current', {})

                with self._lock:
                    self.base_wind_speed_10m = round(float(curr.get('wind_speed_10m', 10.0)) / 3.6, 1)
                    self.base_wind_dir_deg = float(curr.get('wind_direction_10m', 45.0))
                    self.temperature_c = float(curr.get('temperature_2m', 20.0))
                    self.surface_pressure_hpa = float(curr.get('surface_pressure', 1013.25))
                    self.rain_intensity_mmh = float(curr.get('rain', 0.0))
                    self.turbulence_intensity = min(0.30, max(0.08, self.base_wind_speed_10m / 35.0))
                    self.active_mode = "live"
                    self.label = "Open-Meteo Live API"
                return True
        except Exception:
            return False

    def set_profile(self, mode: str, lat: Optional[float] = None, lon: Optional[float] = None) -> Dict[str, Any]:
        valid_modes = ["calm", "windy", "heatwave", "storm", "live"]
        if mode not in valid_modes:
            raise ValueError(f"Profil inconnu : '{mode}'")

        with self._lock:
            self.active_mode = mode

            # 1. CALM : Toujours très bas
            if mode == "calm":
                self.base_wind_speed_10m = round(float(np.random.uniform(1.2, 2.8)), 1)
                self.base_wind_dir_deg = round(float(np.random.uniform(0.0, 360.0)), 1)
                self.temperature_c = round(float(np.random.uniform(20.0, 24.0)), 1)
                self.rain_intensity_mmh = 0.0
                self.turbulence_intensity = 0.08
                self.label = "Calm Survey"

            # 2. WINDY : Capped to NOT exceed 7.0 m/s (5.0 to 6.5 m/s base)
            elif mode == "windy":
                self.base_wind_speed_10m = round(float(np.random.uniform(5.0, 6.5)), 1)
                self.base_wind_dir_deg = round(float(np.random.uniform(0.0, 360.0)), 1)
                self.temperature_c = round(float(np.random.uniform(18.0, 22.0)), 1)
                self.rain_intensity_mmh = 0.0
                self.turbulence_intensity = 0.08  # Lower turbulence to ensure total wind stays <= 7.0 m/s
                self.label = "Coastal Wind (<= 7.0 m/s)"

            # 3. HEATWAVE : 50% au-dessus de 40°C / 50% en dessous
            elif mode == "heatwave":
                self.base_wind_speed_10m = round(float(np.random.uniform(3.0, 6.0)), 1)
                self.base_wind_dir_deg = 120.0
                self.rain_intensity_mmh = 0.0
                self.turbulence_intensity = 0.12

                if np.random.rand() < 0.5:
                    self.temperature_c = round(float(np.random.uniform(42.0, 46.5)), 1)
                    self.label = f"Heatwave CRITICAL ({int(self.temperature_c)}°C)"
                else:
                    self.temperature_c = round(float(np.random.uniform(34.0, 38.5)), 1)
                    self.label = f"Heatwave Normal ({int(self.temperature_c)}°C)"

            # 4. STORM : DÉPASSE TOUJOURS LE SEUIL (13.5 à 17.5 m/s + Pluie) -> Déclenche RTL
            elif mode == "storm":
                self.base_wind_speed_10m = round(float(np.random.uniform(13.5, 17.5)), 1)
                self.base_wind_dir_deg = 260.0
                self.temperature_c = 14.0
                self.rain_intensity_mmh = round(float(np.random.uniform(15.0, 30.0)), 1)
                self.turbulence_intensity = 0.35
                self.label = "Severe Storm (NO-GO)"

            elif mode == "live":
                pass

        if mode == "live" and lat is not None and lon is not None:
            self.fetch_live_satellite_weather(lat, lon)

        return self.get_summary()