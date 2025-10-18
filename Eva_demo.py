from roboticstoolbox.backends import swift
from spatialmath import SE3
import roboticstoolbox as rtb

# ✅ import the class itself, not the package
from Robots.EVABOT.goodbot500 import goodbot500


def main():
    env = swift.Swift()
    env.launch(realtime=True, port=0)

    bot = goodbot500()
    bot.base = SE3(0, 0, 0)        # place the base at world origin
    bot.add_to_env(env)            # add meshes to Swift

    # simple joint-space motion (NO IK, so no 'goal' needed)
    q_start = bot.q[:]             # whatever qtest you set
    q_goal  = [1.0, 0.7, 0.0, 1.2, 1.1, 3.1]
    traj = rtb.jtraj(q_start, q_goal, 150)

    for q in traj.q:
        bot.q = q
        env.step(0.02)

    env.hold()

if __name__ == "__main__":
    main()
