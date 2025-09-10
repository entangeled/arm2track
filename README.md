# arm-to-track

Two robotic arms coordinate to keep a toy train moving:  
- **Arm A (UR3)** pushes the train forward.  
- **Arm B (UR5)** replaces the tracks ahead.  

Built with [Robotics Toolbox for Python](https://github.com/petercorke/roboticstoolbox-python) and Swift visualizer.  

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
