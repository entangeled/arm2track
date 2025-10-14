from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
from swift import Swift
import numpy as np
import time

L1 = RevoluteDH(a=0,     alpha=np.pi/2, d=0.400, qlim=[-np.pi, np.pi])
L2 = RevoluteDH(a=0.250, alpha=0,       d=0,     qlim=[-np.pi/2, np.pi/2])
L3 = RevoluteDH(a=0.250, alpha=0,       d=0,     qlim=[-np.pi, np.pi])
L4 = RevoluteDH(a=0,     alpha=np.pi/2, d=0.300, qlim=[-np.pi, np.pi])
L5 = RevoluteDH(a=0,     alpha=-np.pi/2,d=0,     qlim=[-np.pi, np.pi])
L6 = RevoluteDH(a=0,     alpha=0,       d=0.100, qlim=[-np.pi, np.pi])

kuka_basic = DHRobot([L1, L2, L3, L4, L5, L6], name="KUKA_Basic")


env = Swift()
env.launch()              # open host auto
env.add(kuka_basic)


q_start = np.zeros(6)
q_end   = np.array([0, -np.pi/4, np.pi/3, 0, np.pi/6, 0])

for s in np.linspace(0, 1, 50):
    q = (1 - s) * q_start + s * q_end
    kuka_basic.q = q
    env.step(0.05)

print("Simulation done")
time.sleep(5)
env.close()

