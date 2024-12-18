import rospy

import os.path

import numpy as np

import bot
import generator

class Bot(bot.Bot):
    def __init__(self, asset_root, asset_file):
        bot.Bot.__init__(self, os.path.join(asset_root, asset_file))

    def get_joint_pos(self):
        pass
    
    def get_joint_vel(self):
        pass
    
    def set_joint_target_pos(self, target, wait=False):
        pass

        if wait:
            while max([abs(i-j) for i,j in zip(self.get_joint_pos(), target)]) >= 5e-2:
                yield
    
    def set_joint_target_vel(self, target):
        pass

    def open_gripper(self, wait=False):
        pass

        if wait:
            while max([abs(i-j) for i,j in zip(self.get_joint_pos(), target)]) >= 5e-2:
                yield

    def close_gripper(self, wait=False):
        pass

        if wait:
            while max([abs(i-j) for i,j in zip(self.get_joint_pos(), target)]) >= 5e-2:
                yield

    def point_gripper(self, wait=False):
        pass

        if wait:
            while max([abs(i-j) for i,j in zip(self.get_joint_pos(), target)]) >= 5e-2:
                yield



def generator_delay(time):
    rospy.Duration(time)

stack = [generator.generator_coro(Bot(asset_root, asset_file, gym, env, actor_handle))]

while stack:
    originlen = len(stack)
    for com in stack[-1]:
        if gym.query_viewer_has_closed(viewer):
            stack = [()]
            break

        if com:
            stack.append(generator_delay(com))
            break

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.render_all_camera_sensors(sim)

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        gym.sync_frame_time(sim)

    if len(stack) > originlen:
        pass
    else:
        stack.pop()
