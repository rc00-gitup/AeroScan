"""
Tactical Industrial UAV GCS Server & Photogrammetry Host
=========================================================
- HTTP/REST server for the Ground Control Station (GCS).
- Basic Authentication (engineer / ocp2026).
- Management of 3 navigation modes (one_way, round_trip, infinite).
- Immediate emergency stop command (abort-mission).
- Web interface serving (index.html, track.html, mapping.html).
- Direct in-RAM photogrammetry support (Zero Disk I/O until save).
- Live sensor video stream and progressive seamless orthomosaic.
"""

import base64
import json
import os
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

import planner
import drone_dynamics
from mapping_engine import engine as mapping_engine

USERNAME = "engineer"
PASSWORD = "ocp2026"


class TacticalDroneServer(SimpleHTTPRequestHandler):
    """HTTP request handler for the industrial and photogrammetric GCS."""

    def _send_json_response(self, data: dict, status_code: int = 200):
        """Sends a formatted JSON response with standard headers."""
        payload = json.dumps(data).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(payload)

    def _send_file_response(self, filepath: str, content_type: str):
        """Serves a static file (HTML, CSS, JS, etc.)."""
        if not os.path.exists(filepath):
            self.send_error(404, f"File not found: {filepath}")
            return

        with open(filepath, 'rb') as f:
            content = f.read()

        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-cache, must-revalidate')
        self.end_headers()
        self.wfile.write(content)

    def do_AUTHHEAD(self):
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Industrial UAV GCS Operations"')
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Unauthorized access. Credentials required.")

    def check_auth(self) -> bool:
        """Verifies HTTP Basic authentication."""
        auth_header = self.headers.get("Authorization")
        if auth_header is None:
            self.do_AUTHHEAD()
            return False

        try:
            auth_type, encoded = auth_header.split(" ", 1)
            if auth_type.lower() == "basic":
                decoded = base64.b64decode(encoded.strip()).decode("utf-8")
                user, pwd = decoded.split(":", 1)
                if user == USERNAME and pwd == PASSWORD:
                    return True
        except Exception:
            pass

        self.do_AUTHHEAD()
        return False

    # ==========================================================================
    # GET REQUESTS (TELEMETRY, WEATHER, INTERFACES, RAM MAPPING STREAMS)
    # ==========================================================================

    def do_GET(self):
        if not self.check_auth():
            return

        # 1. Real-time drone telemetry
        if self.path == '/api/drone-telemetry':
            telemetry = drone_dynamics.get_telemetry()
            self._send_json_response(telemetry)
            return

        # 2. Current weather summary
        if self.path.startswith('/api/live-weather'):
            summary = drone_dynamics.weather_env.get_summary()
            self._send_json_response(summary)
            return

        # 3. Active mission retrieval
        if self.path == '/api/get-active-mission':
            mission = drone_dynamics.get_active_mission() or {}
            self._send_json_response(mission)
            return

        # 4. GCS Web interfaces
        if self.path in ['/', '/index.html']:
            self._send_file_response('index.html', 'text/html; charset=utf-8')
            return

        if self.path == '/track.html':
            self._send_file_response('track.html', 'text/html; charset=utf-8')
            return

        if self.path == '/mapping.html':
            self._send_file_response('mapping.html', 'text/html; charset=utf-8')
            return

        # 5. Direct live camera stream (Fixes the broken mini-screen image)
        if self.path.startswith('/api/mapping/live-camera.jpg'):
            frame_bytes = mapping_engine.get_live_camera_bytes()
            if frame_bytes:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(frame_bytes)))
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.end_headers()
                self.wfile.write(frame_bytes)
            else:
                self.send_response(204)
                self.end_headers()
            return

        # 6. Global RAM-assembled orthomosaic stream (Fixes hatched/stitched artifact image)
        if self.path.startswith('/api/mapping/live-ortho'):
            ok, ortho_bytes, meta = mapping_engine.assemble_orthomosaic()
            if ok and ortho_bytes:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(ortho_bytes)))
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.end_headers()
                self.wfile.write(ortho_bytes)
            else:
                self.send_response(204)
                self.end_headers()
            return

        # 7. Photogrammetric status and metadata
        if self.path == '/api/mapping/status':
            status_data = {
                "photos_count": len(mapping_engine.captures),
                "version": mapping_engine.version,
                "latest_capture": mapping_engine.latest_capture,
                "captures": mapping_engine.captures
            }
            self._send_json_response(status_data)
            return

        # 8. Serving of exported static files (e.g., ./static)
        if self.path.startswith('/static/'):
            filepath = "." + self.path.split("?")[0]
            if os.path.exists(filepath):
                ext = filepath.split(".")[-1].lower()
                mime = 'image/jpeg' if ext in ['jpg', 'jpeg'] else 'application/octet-stream'
                self._send_file_response(filepath, mime)
                return

        return super().do_GET()

    # ==========================================================================
    # POST COMMANDS (MISSIONS, WEATHER SELECTION, MAP SAVING)
    # ==========================================================================

    def do_POST(self):
        if not self.check_auth():
            return

        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'

        try:
            data = json.loads(post_data.decode('utf-8')) if post_data else {}
        except Exception as e:
            self._send_json_response({"error": f"Invalid JSON: {str(e)}"}, status_code=400)
            return

        # 1. Immediate emergency stop
        if self.path == '/api/abort-mission':
            drone_dynamics.abort_mission()
            self._send_json_response({"status": "ABORTING_TO_DOCK"})
            return

        # 2. Navigation mode (one_way, round_trip, infinite)
        if self.path == '/api/set-nav-mode':
            mode = data.get('mode', 'one_way')
            drone_dynamics.set_nav_mode(mode)
            self._send_json_response({"nav_mode": mode})
            return

        # 3. In-flight weather profile
        if self.path == '/api/set-weather-mode':
            mode = data.get('mode', 'live')
            try:
                updated_summary = drone_dynamics.set_weather_profile(mode)
                active_mission = drone_dynamics.get_active_mission()
                has_active_flight = active_mission is not None and "raw_polygon" in active_mission
                self._send_json_response({
                    "weather": updated_summary,
                    "replanning_in_progress": has_active_flight,
                    "active_mission": active_mission
                })
            except ValueError as err:
                self._send_json_response({"error": str(err)}, status_code=400)
            return

        # 4. Save orthomosaic map to disk (on demand)
        if self.path == '/api/mapping/save-map':
            os.makedirs("./static", exist_ok=True)
            export_filename = f"orthomosaic_{int(time.time())}.jpg"
            export_path = os.path.join("./static", export_filename)
            ok, msg = mapping_engine.save_current_orthomosaic(export_path)
            if ok:
                self._send_json_response({
                    "success": True,
                    "file_url": f"/static/{export_filename}",
                    "filename": export_filename
                })
            else:
                self._send_json_response({"success": False, "error": msg}, status_code=400)
            return

        # 5. Reset mapping memory
        if self.path == '/api/mapping/reset':
            mapping_engine.reset_mission()
            self._send_json_response({"status": "RESET_OK"})
            return

        # 6. Define and validate a new mission
        if self.path == '/api/set-mission':
            try:
                polygon_pts = [(float(pt['lat']), float(pt.get('lon', pt.get('lng')))) for pt in data['polygon']]
                dock_station = data.get('dock_station', {'lat': 32.2515, 'lng': -9.2630})
                nav_mode = data.get('mode', 'one_way')

                dock_lat = float(dock_station["lat"])
                dock_lon = float(dock_station.get("lon", dock_station.get("lng", 0.0)))

                # Reset photogrammetric memory for the new flight
                mapping_engine.reset_mission()

                # Fetch weather
                drone_dynamics.weather_env.fetch_live_satellite_weather(dock_lat, dock_lon, timeout_sec=1.5)
                w_sum = drone_dynamics.weather_env.get_summary()

                # Calculate flight plan with 100% autonomous altitude and NFZ handling
                mission_plan = planner.plan_mission(
                    polygon_coords=polygon_pts,
                    dock_station=dock_station,
                    wind_speed_ms=w_sum["wind_speed_ms"],
                    wind_dir_deg=w_sum["wind_direction_deg"],
                    air_density=drone_dynamics.weather_env.get_air_density(altitude_m=35.0),
                    auto_fetch_osm=True
                )

                # Energy check
                feasibility = mission_plan.get("feasibility", {})
                if not feasibility.get("is_feasible", False):
                    self._send_json_response({
                        "error": "Mission impossible: area too large for current autonomy.",
                        "feasibility": feasibility
                    }, status_code=400)
                    return

                if not w_sum.get("is_flight_permitted", True):
                    self._send_json_response({
                        "error": "Mission REJECTED: Weather conditions outside operational limits.",
                        "weather": w_sum
                    }, status_code=400)
                    return

                # MAJOR OPTIMIZATION: Immediate pre-loading of tiles into RAM
                # Eliminates any freeze or frame loss at the start of the flight
                mapping_engine.preload_flight_area(mission_plan.get("waypoints", []))

                # Start autonomous flight
                drone_dynamics.start_mission(mission_plan, dock_station, mode=nav_mode)
                self._send_json_response(mission_plan, status_code=200)

            except Exception as e:
                self._send_json_response({"error": f"Error during planning: {str(e)}"}, status_code=500)
            return

        self.send_error(404, "Endpoint not recognized.")


# ==============================================================================
# SERVER ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    os.makedirs("./static", exist_ok=True)
    server_address = ("", 8000)
    httpd = HTTPServer(server_address, TacticalDroneServer)
    print("🚀 [GCS Server] Tactical Drone Server STARTED at http://localhost:8000")
    print("🔒 Authentication enabled: user 'engineer'")
    print("🗺️ Photogrammetry Engine ready: In-RAM Dynamic Staging active.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 [GCS Server] Server shutdown requested.")
        drone_dynamics.weather_env.stop()
        httpd.server_close()
        print("✅ System safely stopped.")