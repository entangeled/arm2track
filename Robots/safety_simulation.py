# safety_simulation.py
# Full robot safety simulation packaged into a class for import/use in main.py

import time
import threading
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from math import pi
import numpy as np
import roboticstoolbox as rtb
import swift
from spatialmath import SE3

try:
    import spatialgeometry as sg
except Exception:
    sg = None


# ---------------- Utility Classes ----------------

class AxisAlignedBox:
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, name="zone"):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.zmin, self.zmax = zmin, zmax
        self.name = name

    def contains(self, p):
        x, y, z = p
        return (self.xmin <= x <= self.xmax and
                self.ymin <= y <= self.ymax and
                self.zmin <= z <= self.zmax)


class MovingObstacle:
    def __init__(self, pos, vel, radius=0.06):
        self.p = np.array(pos, dtype=float)
        self.v = np.array(vel, dtype=float)
        self.r = float(radius)
        self.node = None

    def step(self, dt):
        self.p = self.p + self.v * dt
        if self.node is not None:
            self.node.T = SE3(*self.p.tolist())


# ---------------- Main Simulation Class ----------------

class SafetySimulation:
    def __init__(self, light_curtain_x=0.35):
        self.light_curtain_x = float(light_curtain_x)
        self.estop_latched = False
        self.paused = False
        self.quit = False
        self.speed_scale = 1.0
        self.curtain_on = True
        self.fences = []
        self.obstacles = []
        self._server = None
        self._thread = None
        self.env = None
        self.bot = None
        self.nodes = {}

    # ---------- E-stop and safety logic ----------
    def gui_estop(self):
        print("🔴 GUI Emergency Stop pressed")
        self.estop_latched = True

    def hardware_estop(self):
        print("🔴 Hardware Emergency Stop triggered")
        self.estop_latched = True

    def reset(self):
        print("✅ Resetting: E-Stop cleared")
        self.estop_latched = False
        self.paused = True

    def resume(self):
        if self.estop_latched:
            print("⚠️  Cannot resume: E-Stop latched")
            return
        print("▶️  Resuming from protective stop")
        self.paused = False

    def light_curtain_trip(self, x):
        if self.curtain_on and x >= self.light_curtain_x:
            print(f"🟡 Light curtain tripped at x={self.light_curtain_x:.2f}")
            self.paused = True
            return True
        return False

    def fence_breach(self, p):
        for z in self.fences:
            if z.contains(p):
                print(f"🟠 Fence intrusion: {z.name}")
                self.paused = True
                return True
        return False

    def predict_collision(self, robot_p, robot_v, horizon=1.5, dt=0.05):
        if not self.obstacles:
            return False
        rp, rv = np.array(robot_p), np.array(robot_v)
        for obs in self.obstacles:
            for t in np.arange(0.0, horizon, dt):
                pr = rp + rv * t
                po = obs.p + obs.v * t
                if np.linalg.norm(pr - po) <= (0.08 + obs.r):
                    print("🧨 Predicted collision!")
                    self.paused = True
                    return True
        return False

    # ---------- HTTP GUI ----------
    def start_http(self, host="127.0.0.1", port=8765):
        sim = self

        class Panel(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/cmd":
                    q = urllib.parse.parse_qs(parsed.query)
                    if "e" in q: sim.gui_estop()
                    if "r" in q: sim.reset()
                    if "u" in q: sim.resume()
                    if "p" in q: sim.paused = not sim.paused
                    if "c" in q: sim.curtain_on = not sim.curtain_on
                    if "s" in q:
                        try:
                            sim.speed_scale = float(q["s"][0])
                            sim.speed_scale = max(0.1, min(2.0, sim.speed_scale))
                        except Exception:
                            pass
                    if "spawn" in q: sim.spawn_obstacle()
                    self._redir("/")
                    return
                self._html(self._page())

            def log_message(self, *a, **k): return

            def _redir(self, where):
                self.send_response(302)
                self.send_header("Location", where)
                self.end_headers()

            def _html(self, body):
                b = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def _page(self):
                est = "ON" if sim.estop_latched else "OFF"
                pau = "ON" if sim.paused else "OFF"
                cur = "ON" if sim.curtain_on else "OFF"
                return f"""
<!doctype html><html><head><meta charset="utf-8"/><title>Safety Panel</title></head>
<body style="font-family:sans-serif;margin:24px">
<h2>Safety Panel</h2>
<p>E-Stop: {est} | Paused: {pau} | Curtain: {cur} | Speed: {sim.speed_scale:.2f}</p>
<a href="/cmd?e=1">🔴 E-Stop</a> |
<a href="/cmd?r=1">Reset</a> |
<a href="/cmd?u=1">Resume</a> |
<a href="/cmd?p=1">Pause</a> |
<a href="/cmd?c=1">Toggle Curtain</a> |
<a href="/cmd?spawn=1">Spawn Obstacle</a>
</body></html>"""

        self._server = HTTPServer((host, port), Panel)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"🌐 Safety Panel available at http://{host}:{port}")

    def stop_http(self):
        try:
            if self._server:
                self._server.shutdown()
        finally:
            self._server = None
            self._thread = None

    # ---------- Environment setup ----------
    def setup_environment(self):
        self.env = swift.Swift()
        self.env.launch(realtime=True)
        self.bot = rtb.models.Panda()
        self.env.add(self.bot)

        self.fences = [AxisAlignedBox(-2.5, 2.5, -3.0, 3.0, 0.0, 0.6, "Perimeter")]

        if sg:
            front = sg.Cuboid(scale=[5.0, 0.05, 0.6], pose=SE3(0.0, 3.0, 0.3), color=[0.9, 0.1, 0.1, 0.4])
            back = sg.Cuboid(scale=[5.0, 0.05, 0.6], pose=SE3(0.0, -3.0, 0.3), color=[0.9, 0.1, 0.1, 0.4])
            left = sg.Cuboid(scale=[0.05, 6.0, 0.6], pose=SE3(-2.5, 0.0, 0.3), color=[0.9, 0.1, 0.1, 0.4])
            right = sg.Cuboid(scale=[0.05, 6.0, 0.6], pose=SE3(2.5, 0.0, 0.3), color=[0.9, 0.1, 0.1, 0.4])
            floor = sg.Cuboid(scale=[5.0, 6.0, 0.02], pose=SE3(0.0, 0.0, -0.01), color=[0.3, 0.3, 0.3, 0.6])
            for n in [front, back, left, right, floor]:
                self.env.add(n)

            curtain = sg.Cuboid(scale=[0.008, 6.0, 0.6],
                                pose=SE3(self.light_curtain_x, 0.0, 0.3),
                                color=[1.0, 0.9, 0.1, 0.4])
            self.env.add(curtain)
            self.nodes["curtain"] = curtain
        else:
            print("⚠️ spatialgeometry not installed, no fence visuals")

    def spawn_obstacle(self):
        obs = MovingObstacle([0.45, 0.0, 0.12], [-0.12, 0.0, 0.0])
        self.obstacles = [obs]
        if sg and self.env:
            sphere = sg.Sphere(radius=0.06, pose=SE3(*obs.p.tolist()), color=[0.2, 0.5, 0.95, 0.9])
            self.env.add(sphere)
            obs.node = sphere
        print("🟦 Spawned moving obstacle.")

    # ---------- Main robot loop ----------
    def run_trajectory(self):
        print("▶️ Starting trajectory...")
        tcp = np.array([0.15, 0.0, 0.15])
        self.bot.q = [0.0, -pi/4, 0.0, -pi/2, 0.0, pi/3, 0.0]
        dt = 0.05
        v_cmd = np.array([0.02, 0.0, 0.0])

        while not self.quit:
            if self.estop_latched:
                print("[SAFETY] E-Stop latched → stop loop")
                break
            if self.paused:
                time.sleep(0.2)
                continue

            tcp = tcp + v_cmd * self.speed_scale
            sol = self.bot.ikine_LM(SE3(*tcp.tolist()))
            if sol.success:
                self.bot.q = sol.q
            else:
                print("IK failed → stopping.")
                break

            self.env.step(dt)

            # Safety conditions
            if self.light_curtain_trip(tcp[0]): continue
            if self.fence_breach(tcp): continue
            if self.predict_collision(tcp, v_cmd * self.speed_scale): continue

            time.sleep(dt)

    # ---------- Public interface ----------
    def start(self):
        """Launch environment and start sim loop."""
        self.setup_environment()
        self.start_http()
        self.spawn_obstacle()

        t = threading.Thread(target=self.run_trajectory, daemon=True)
        t.start()
        try:
            while t.is_alive():
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.hardware_estop()
            self.quit = True
        finally:
            self.stop_http()
            print("Simulation ended.")


if __name__ == "__main__":
    sim = SafetySimulation()
    sim.start()
