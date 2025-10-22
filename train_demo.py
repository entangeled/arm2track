# train_demo_goodbot.py — Ring + Train + Panda follow + Animated Segment Toggles + goodbot500 pick/place
from math import pi, cos
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from spatialmath import SE3
from spatialgeometry import Mesh, Cuboid
from roboticstoolbox.backends import swift
from roboticstoolbox import models  # Panda model
import roboticstoolbox as rtb

# ← your custom EVABOT
from Robots.EVABOT.barbie import barbie


# <- safety feature code IF CODE ERRORS REMOVE THIS!!
from safety_simulation import SafetySimulation



# =========================
# Config
# =========================
@dataclass
class RingCfg:
    n_segments: int = 9
    radius: float = 0.8
    height_z: float = 0.10
    yaw_tweak_deg: float = 69.65
    center_x: float = 0.0
    center_y: float = 0.0
    global_yaw_deg: float = 0.0
    unit_scale: float = 0.0003
    color: tuple = (0.35, 0.35, 0.38, 1.0)

# -------------------------------
# Track ring
# -------------------------------
class TrackRing:
    def __init__(self, env, segment_path, cfg: RingCfg):
        self.env = env; self.cfg = cfg
        self.segment_path = segment_path
        s = cfg.unit_scale; self.scale = [s, s, s]
        self.pieces = []; self.gapset = set()
        self.out_offset = 0.30
        self.anim = {}; self.missing_slots = {5}
        self.slot_poses = [None]*cfg.n_segments

    def build(self):
        n = self.cfg.n_segments
        for i in range(n):
            theta = 2*pi*i/n
            T = (SE3(self.cfg.center_x, self.cfg.center_y, self.cfg.height_z)
                * SE3.Rz(theta) * SE3.Tx(self.cfg.radius)
                * SE3.Rz(pi/2 + self.cfg.yaw_tweak_deg*pi/180)
                * SE3.Rx(-pi/2))
            self.slot_poses[i] = T
            if i in self.missing_slots:
                self.pieces.append(None)
            else:
                m = Mesh(str(self.segment_path), pose=SE3(), scale=self.scale, color=self.cfg.color)
                self.env.add(m); self.pieces.append(m)
        self.gapset |= set(self.missing_slots)
        self.update()


    def _place_segment(self, i:int, r:float):
        if self.pieces[i] is None: return
        c=self.cfg; theta=2*pi*i/c.n_segments
        T=(SE3(c.center_x,c.center_y,c.height_z)
           * SE3.Rz(theta) * SE3.Tx(r)
           * SE3.Rz(pi/2 + c.yaw_tweak_deg*pi/180)
           * SE3.Rx(-pi/2))
        self.pieces[i].T = T.A

    def update(self):
        c=self.cfg; r_in,r_out=c.radius,c.radius+self.out_offset
        done=[]
        for i in range(c.n_segments):
            if i in self.missing_slots: self.gapset.add(i); continue
            if i in self.anim:
                st=self.anim[i]; st['t']+=0.05
                u=min(1,st['t']/st['dur']); s=0.5-0.5*cos(pi*u)
                r=st['r0']+(st['r1']-st['r0'])*s
                self._place_segment(i,r)
                if u>=1: done.append(i)
            else:
                self._place_segment(i,r_out if i in self.gapset else r_in)
        for i in done:
            final_r=self.anim[i]['r1']
            if abs(final_r-(c.radius+self.out_offset))<1e-9: self.gapset.add(i)
            else: self.gapset.discard(i)
            del self.anim[i]

    def start_slide(self, idx:int, present:bool, dur:float=0.8):
        if idx in self.missing_slots: return
        c=self.cfg; r_in,r_out=c.radius,c.radius+self.out_offset
        r_now=r_out if idx in self.gapset else r_in
        r_tar=r_in if present else r_out
        self.anim[idx]={'t':0,'dur':dur,'r0':r_now,'r1':r_tar}

    def allowed_speed(self,theta_now:float,v:float)->float:
        n=self.cfg.n_segments
        gaps=self.gapset|self.missing_slots
        i=int(((theta_now%(2*pi))/(2*pi))*n)%n; ahead=(i+1)%n
        return 0.0 if (i in gaps or ahead in gaps) else v

# -------------------------------
# Train
# -------------------------------
class Train:
    """Train perfectly aligned on the track and driven by angular speed."""
    def __init__(self, env, mesh_path, cfg: RingCfg):
        self.env = env
        self.cfg = cfg
        self.theta = 0.0
        self.speed = 1.0  # rad/s
        s = cfg.unit_scale

        self.mesh = Mesh(str(mesh_path), pose=SE3(), scale=[s, s, s], color=[0.12, 0.12, 0.12, 1])
        env.add(self.mesh)

        self.height_offset = 0.10
        self.radial_offset = 0.11
        self.forward_offset = 0.00
        self.roll_deg = 0.0
        self.pitch_deg = 84.0
        self.yaw_deg = 180.0

    def pose_on_ring(self, theta):
        c = self.cfg
        T = (
            SE3(c.center_x, c.center_y, c.height_z)
            * SE3.Rz(c.global_yaw_deg * pi / 180)
            * SE3.Rz(theta)
            * SE3.Tx(c.radius)
            * SE3.Rz(pi / 2 + c.yaw_tweak_deg * pi / 180)
            * SE3.Rx(-pi / 2)
        )
        return (
            T
            * SE3.Tz(self.height_offset)
            * SE3.Tx(self.forward_offset)
            * SE3.Ty(self.radial_offset)
            * SE3.Rx(self.roll_deg * pi / 180)
            * SE3.Ry(self.pitch_deg * pi / 180)
            * SE3.Rz(self.yaw_deg * pi / 180)
        )

    def step(self, dt, speed_override=None):
        v = self.speed if speed_override is None else speed_override
        self.theta = (self.theta + v * dt) % (2 * pi)
        self.mesh.T = self.pose_on_ring(self.theta).A

# -------------------------------
# Panda follow
# -------------------------------
class RobotArm:
    def __init__(self, env, cfg):
        self.env=env; self.cfg=cfg
        self.robot=models.Panda()
        self.robot.base=SE3(cfg.center_x,cfg.center_y,0)
        env.add(self.robot)
        self.q_follow=self.robot.qz.copy(); self.q_follow[1]=0.5; self.q_follow[2]=0.4
        self.q_follow[3]=-1.7; self.q_follow[5]=2.1; self.q_follow[6]=0.7

    def follow_train_yaw(self,theta):
        q=self.q_follow.copy(); q[0]=theta; self.robot.q=q

# -------------------------------
# App
# -------------------------------
class App:
    def __init__(self):
        self.env=swift.Swift(); self.env.launch(realtime=True,port=0); self.dt=0.05

    def add_floor(self): self.env.add(Cuboid(scale=[8,8,0.02],pose=SE3(0,0,-0.01),color=[0.9,0.9,0.95,1]))
    def add_ring(self,seg_path,cfg): self.ring=TrackRing(self.env,seg_path,cfg); self.ring.build()
    def add_train(self,train_path,cfg): self.train=Train(self.env,train_path,cfg)
    def add_robot(self,cfg): self.robot=RobotArm(self.env,cfg)

    # only goodbot500
    def add_goodbot(self,cfg):
        self.good=barbie()
        theta=2*pi*5/cfg.n_segments
        self.good.base=(SE3(cfg.center_x,cfg.center_y,0)
                        * SE3.Rz(theta)
                        * SE3.Tx(cfg.radius+0.12)
                        * SE3.Rz(pi))
        self.good.add_to_env(self.env)

    def loop(self):
        try:
            while True:
                v=self.ring.allowed_speed(self.train.theta,self.train.speed)
                self.train.step(self.dt,v)
                self.ring.update()
                self.robot.follow_train_yaw(self.train.theta)
                self.env.step(self.dt)
        except KeyboardInterrupt:
            print("Exiting simulation...")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    seg_path = here / "track_segment.STL"
    train_path = here / "train.STL"

    cfg = RingCfg()
    app = App()
    app.add_floor()
    app.add_ring(seg_path, cfg)
    app.add_train(train_path, cfg)
    app.add_robot(cfg)
    app.add_goodbot(cfg)

    # Optional: simple joint-space motion demo for goodbot500 (no zones needed)
    def wiggle_goodbot(e=None):
        q0 = app.good.q[:] if hasattr(app.good, "q") else [0, 0, 0, 0, 0, 0]
        q1 = [q0[0]+0.6, q0[1]+0.4, q0[2]-0.5, q0[3]+0.6, q0[4]-0.4, q0[5]]
        traj = rtb.jtraj(q0, q1, 120)
        for q in traj.q:
            app.good.q = q
            app.env.step(app.dt)

    app.env.add(swift.Button(cb=wiggle_goodbot, desc="goodbot500: wiggle"))

    app.loop()
