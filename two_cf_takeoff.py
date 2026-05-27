from crazyflie_py import Crazyswarm
import time

swarm = Crazyswarm()
timeHelper = swarm.timeHelper

allcfs = swarm.allcfs

print("Taking off...")
allcfs.takeoff(targetHeight=1.0, duration=3.0)
timeHelper.sleep(4.0)

print("Hovering...")
timeHelper.sleep(3.0)

print("Landing...")
allcfs.land(targetHeight=0.04, duration=3.0)
timeHelper.sleep(4.0)

print("Done.")
