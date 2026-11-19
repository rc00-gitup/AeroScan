# AeroScan
### Autonomous Drone Mapping & Aerial Inspection System

**Turning dangerous inspections into autonomous aerial operations.**

AeroScan is an autonomous drone mapping and aerial inspection system designed to improve safety and efficiency in industrial environments.

The project aims to reduce human exposure to hazardous working conditions by enabling autonomous aerial surveys, intelligent flight planning, real-time monitoring, and progressive aerial map generation.

---

## Overview

Industrial environments such as mines, quarries, construction sites, and factory rooftops often require inspections in areas that present significant safety risks.

AeroScan addresses this challenge through an autonomous aerial inspection platform that combines flight planning, environmental monitoring, drone simulation, and real-time image processing.

The objective is simple: allow operators to plan an inspection, monitor the mission, and obtain a detailed aerial map without having to enter dangerous areas.

## Key Features

### 1. Autonomous Mission Planning
- Define an inspection area through an interactive interface.
- Generate optimized flight paths based on the selected area.
- Support different navigation modes.
- Visualize mission trajectories.

### 2. Intelligent Flight Monitoring
- Monitor drone telemetry.
- Track battery status.
- Monitor environmental and weather conditions.
- Support emergency mission abortion.

### 3. Real-Time Aerial Mapping
- Capture aerial images during the mission.
- Process captured images in RAM.
- Progressively generate an orthomosaic.
- Preview the resulting map during the mission.
- Export the final map when needed.

### 4. Ground Control Station (GCS)
- Interactive mission management interface.
- Live drone telemetry visualization.
- Mission tracking.
- Weather monitoring.
- Mapping visualization and export.

---

## System Architecture

AeroScan is organized into several interconnected modules:

| Module | Responsibility |
|---|---|
| `server.py` | HTTP server, API endpoints and Ground Control Station |
| `planner.py` | Flight path planning |
| `drone_dynamics.py` | Drone dynamics and telemetry |
| `battery_model.py` | Battery behavior and monitoring |
| `weather_engine.py` | Weather environment management |
| `mapping_engine.py` | Image processing and orthomosaic generation |
| `index.html` | Main control interface |
| `track.html` | Flight tracking interface |
| `mapping.html` | Aerial mapping interface |

## Project Structure

```text
AeroScan/
│
├── server.py
├── planner.py
├── drone_dynamics.py
├── battery_model.py
├── weather_engine.py
├── mapping_engine.py
│
├── index.html
├── track.html
├── mapping.html
│
├── photos_drone/
├── tile_cache/
├── static/
│
└── test_stitching_qa.py
```

## Technologies

- **Python** — Backend logic and system management.
- **HTML, CSS and JavaScript** — User interfaces.
- **Image Processing** — Aerial image processing and orthomosaic generation.
- **REST API** — Communication between the control interface and backend.

## Project Goals

AeroScan aims to:

- Reduce human exposure to dangerous inspection environments.
- Automate aerial survey operations.
- Improve inspection efficiency.
- Generate high-resolution aerial maps.
- Provide operators with a centralized mission monitoring interface.

## Development Status

AeroScan is under development.

The project focuses on developing and integrating the core components of an autonomous aerial inspection platform.

Features and capabilities are progressively implemented and tested.

## Future Improvements

- Integration with real drone flight controllers.
- GPS-based autonomous navigation.
- Advanced obstacle detection and avoidance.
- Improved photogrammetry and map reconstruction.
- Enhanced battery prediction algorithms.
- Deployment on real industrial inspection scenarios.

---

## Safety Notice

AeroScan is a development project. Simulation and software capabilities do not constitute certification for autonomous flight or industrial deployment.

Real-world operations require appropriate hardware integration, flight testing, safety validation, and regulatory compliance.

---

**AeroScan — Making industrial inspections safer through autonomous aerial systems.**