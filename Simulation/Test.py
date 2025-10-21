import numpy as np
import roboticstoolbox as rtb

panda = rtb.models.Panda()
q0 = panda.qz
q1 = q0.copy(); q1[0] += np.deg2rad(30)

traj = rtb.jtraj(q0, q1, 50)

print("q shape:", traj.q.shape)
print("qd shape:", traj.qd.shape)
print("qdd shape:", traj.qdd.shape)
