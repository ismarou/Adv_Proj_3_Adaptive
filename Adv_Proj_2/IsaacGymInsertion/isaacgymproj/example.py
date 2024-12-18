import sys
import os.path

import numpy as np
import imageio
from isaacgym import gymapi

import bot
import generator

class Bot(bot.Bot):
    def __init__(self, asset_root, asset_file, gym, env, actor):
        bot.Bot.__init__(self, os.path.join(asset_root, asset_file))
        self.gym = gym
        self.env = env
        self.actor = actor

        props = self.gym.get_actor_dof_properties(self.env, self.actor)
        props['driveMode'].fill(gymapi.DOF_MODE_POS)
        props['stiffness'].fill(900.0)
        self.gym.set_actor_dof_properties(self.env, self.actor, props)
        
    
    def get_joint_pos(self):
        return [it for it in self.gym.get_actor_dof_states(self.env, self.actor, gymapi.STATE_POS)['pos']]
    
    def get_joint_vel(self):
        return [it for it in self.gym.get_actor_dof_states(self.env, self.actor, gymapi.STATE_VEL)['vel']]
    
    def set_joint_target_pos(self, target, wait=False):
        props = self.gym.get_actor_dof_properties(self.env, self.actor)
        props['driveMode'][0:7] = gymapi.DOF_MODE_POS
        props['stiffness'][0:7] = 900.0
        props['damping'][0:7] = 100.0
        self.gym.set_actor_dof_properties(self.env, self.actor, props)
        t = np.array(target).astype('f')
        targets = self.gym.get_actor_dof_position_targets(self.env, self.actor)
        targets[0:7] = t
        self.gym.set_actor_dof_position_targets(self.env, self.actor, targets)

        if wait:
            while max([abs(i-j) for i,j in zip(self.get_joint_pos(), target)]) >= 5e-2:
                yield
    
    def set_joint_target_vel(self, target):
        props = self.gym.get_actor_dof_properties(self.env, self.actor)
        props['driveMode'][0:7] = gymapi.DOF_MODE_VEL
        props['stiffness'][0:7] = 900.0
        props['damping'][0:7] = 100.0
        self.gym.set_actor_dof_properties(self.env, self.actor, props)
        t = np.array(target).astype('f')
        self.gym.set_actor_dof_velocity_targets(self.env, self.actor, t)

    def open_gripper(self, wait=False):
        props = self.gym.get_actor_dof_properties(self.env, self.actor)
        for i in [8, 12, 15]:
            props['driveMode'][i] = gymapi.DOF_MODE_POS
            props['stiffness'][i] = 900.0
            props['damping'][i] = 100.0
        self.gym.set_actor_dof_properties(self.env, self.actor, props)

        targets = self.gym.get_actor_dof_position_targets(self.env, self.actor)
        for i in [8, 12, 15]:
            targets[i] = 0
        self.gym.set_actor_dof_position_targets(self.env, self.actor, targets)

        if wait:
            while max([abs(i-j) for i,j in zip(self.get_joint_pos(), target)]) >= 5e-2:
                yield

    def close_gripper(self, wait=False):
        props = self.gym.get_actor_dof_properties(self.env, self.actor)
        for i in [8, 12, 15]:
            props['driveMode'][i] = gymapi.DOF_MODE_POS
            props['stiffness'][i] = 900.0
            props['damping'][i] = 100.0
        self.gym.set_actor_dof_properties(self.env, self.actor, props)

        targets = self.gym.get_actor_dof_position_targets(self.env, self.actor)
        for i in [8, 12, 15]:
            targets[i] = 1
        self.gym.set_actor_dof_position_targets(self.env, self.actor, targets)

        if wait:
            while max([abs(i-j) for i,j in zip(self.get_joint_pos(), target)]) >= 5e-2:
                yield

    def point_gripper(self, wait=False):
        props = self.gym.get_actor_dof_properties(self.env, self.actor)
        for i in [8,9, 12,13, 15,16,17]:
            props['driveMode'][i] = gymapi.DOF_MODE_POS
            props['stiffness'][i] = 900.0
            props['damping'][i] = 100.0
        self.gym.set_actor_dof_properties(self.env, self.actor, props)

        targets = self.gym.get_actor_dof_position_targets(self.env, self.actor)
        targets[8] = 1
        targets[9] = 1
        targets[12] = 1
        targets[13] = 1
        targets[15] = 0.5
        targets[16] = 0
        targets[17] = -0.3
        self.gym.set_actor_dof_position_targets(self.env, self.actor, targets)

        if wait:
            while max([abs(i-j) for i,j in zip(self.get_joint_pos(), targets)]) >= 5e-2:
                yield



gym = gymapi.acquire_gym()

# get default set of parameters
sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

# set Flex-specific parameters
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.5

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

asset_root = 'assets'
asset_file = 'factory/urdf/factory_kuka.urdf'

assetoptions = gymapi.AssetOptions()
assetoptions.fix_base_link = True
asset = gym.load_asset(sim, asset_root, asset_file, assetoptions)
print(gym.get_asset_dof_properties(asset))

assetoptions2 = gymapi.AssetOptions()
asset2 = gym.load_asset(sim, asset_root, 'factory/urdf/banana.urdf', assetoptions2)

assetoptions3 = gymapi.AssetOptions()
assetoptions3.fix_base_link = True
asset3 = gym.load_asset(sim, asset_root, 'box.urdf', assetoptions3)

def generator_delay(time):
    while time > 0:
        time -= sim_params.dt
        yield

def run_headless(gen, outvidpath):
    spacing = 2.0
    lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    upper = gymapi.Vec3(spacing, spacing, spacing)

    env = gym.create_env(sim, lower, upper, 8)


    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(10.0, 5.0, 0.0)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor_handle = gym.create_actor(env, asset, pose, "MyActor", 0, 1)

    pose2 = gymapi.Transform()
    pose2.p = gymapi.Vec3(10.5, 5.0, 0.0)
    pose2.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor_handle2 = gym.create_actor(env, asset2, pose2, "MyActor2", 0, 0)


    pose3 = gymapi.Transform()
    pose3.p = gymapi.Vec3(10.5, 5.0, 0.0)
    pose3.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor_handle3 = gym.create_actor(env, asset3, pose3, "MyActor3", 0, 0)
    stack = [gen(Bot(asset_root, asset_file, gym, env, actor_handle))]

    cam_props = gymapi.CameraProperties()

    cam_props.width = 1920
    cam_props.height = 1080
    camera_handle = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(camera_handle, env, gymapi.Vec3(11.5, 5, 1), gymapi.Vec3(10.4, 5, 0))


    w = imageio.get_writer(outvidpath, fps=60, macro_block_size=None)
    while stack:
        originlen = len(stack)
        for com in stack[-1]:
            if com:
                stack.append(generator_delay(com))
                break

            gym.simulate(sim)
            gym.fetch_results(sim, True)

            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)

            w.append_data(gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR).reshape(1080,1920,4))

        if len(stack) > originlen:
            pass
        else:
            stack.pop()

    w.close()
    gym.destroy_env(env)


def run_demo(gen):
    spacing = 2.0
    lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    upper = gymapi.Vec3(spacing, spacing, spacing)

    env = gym.create_env(sim, lower, upper, 8)


    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(10.0, 5.0, 0.0)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor_handle = gym.create_actor(env, asset, pose, "MyActor", 0, 1)

    pose2 = gymapi.Transform()
    pose2.p = gymapi.Vec3(10.5, 5.0, 0.0)
    pose2.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor_handle2 = gym.create_actor(env, asset2, pose2, "MyActor2", 0, 0)


    pose3 = gymapi.Transform()
    pose3.p = gymapi.Vec3(10.5, 5.0, 0.0)
    pose3.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor_handle3 = gym.create_actor(env, asset3, pose3, "MyActor3", 0, 0)
    stack = [gen(Bot(asset_root, asset_file, gym, env, actor_handle))]

    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)

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

            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)

            gym.sync_frame_time(sim)

        if len(stack) > originlen:
            pass
        else:
            stack.pop()

    
    gym.destroy_env(env)
    gym.destroy_viewer(viewer)


def run_headlesstarget(gen, outvidpath):
    spacing = 2.0
    lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    upper = gymapi.Vec3(spacing, spacing, spacing)

    env = gym.create_env(sim, lower, upper, 8)


    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(10.0, 5.0, 0.0)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor_handle = gym.create_actor(env, asset, pose, "MyActor", 0, 1)

    pose2 = gymapi.Transform()
    pose2.p = gymapi.Vec3(10.5, 5.0, 0.0)
    pose2.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor_handle2 = gym.create_actor(env, asset2, pose2, "MyActor2", 0, 0)


    pose3 = gymapi.Transform()
    pose3.p = gymapi.Vec3(10.5, 5.0, 0.0)
    pose3.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor_handle3 = gym.create_actor(env, asset3, pose3, "MyActor3", 0, 0)
    stack = [gen(Bot(asset_root, asset_file, gym, env, actor_handle), np.random.random((2,)),np.random.random((2,)))]

    cam_props = gymapi.CameraProperties()

    cam_props.width = 1920
    cam_props.height = 1080
    camera_handle = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(camera_handle, env, gymapi.Vec3(11.5, 5, 1), gymapi.Vec3(10.4, 5, 0))


    w = imageio.get_writer(outvidpath, fps=60, macro_block_size=None)
    while stack:
        originlen = len(stack)
        for com in stack[-1]:
            if com:
                stack.append(generator_delay(com))
                break

            gym.simulate(sim)
            gym.fetch_results(sim, True)

            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)

            w.append_data(gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR).reshape(1080,1920,4))

        if len(stack) > originlen:
            pass
        else:
            stack.pop()

    w.close()
    gym.destroy_env(env)

def run_headlesstarget2(gen, outvidpath, w):
    spacing = 2.0
    lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    upper = gymapi.Vec3(spacing, spacing, spacing)

    env = gym.create_env(sim, lower, upper, 8)


    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(10.0, 5.0, 0.0)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor_handle = gym.create_actor(env, asset, pose, "MyActor", 0, 1)

    pose2 = gymapi.Transform()
    pose2.p = gymapi.Vec3(10.5, 5.0, 0.0)
    pose2.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor_handle2 = gym.create_actor(env, asset2, pose2, "MyActor2", 0, 0)


    pose3 = gymapi.Transform()
    pose3.p = gymapi.Vec3(10.5, 5.0, 0.0)
    pose3.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor_handle3 = gym.create_actor(env, asset3, pose3, "MyActor3", 0, 0)
    stack = [gen(Bot(asset_root, asset_file, gym, env, actor_handle), np.random.random((2,)),np.random.random((2,)))]

    cam_props = gymapi.CameraProperties()

    cam_props.width = 1920
    cam_props.height = 1080
    camera_handle = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(camera_handle, env, gymapi.Vec3(11.5, 5, 1), gymapi.Vec3(10.4, 5, 0))

    while stack:
        originlen = len(stack)
        for com in stack[-1]:
            if com:
                stack.append(generator_delay(com))
                break

            gym.simulate(sim)
            gym.fetch_results(sim, True)

            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)

            w.append_data(gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR).reshape(1080,1920,4))

        if len(stack) > originlen:
            pass
        else:
            stack.pop()

    gym.destroy_camera_sensor(sim, env, camera_handle)
    gym.destroy_env(env)


if sys.argv[1] == 'headless':
    run_headless(generator.generator_coro, 'vid.mp4')
elif sys.argv[1] == 'random':
    run_headlesstarget(generator.generator_coro2, sys.argv[2])
elif sys.argv[1] == 'random2':
    w = imageio.get_writer(sys.argv[2], fps=60, macro_block_size=None)
    for i in range(3):
        run_headlesstarget2(generator.generator_coro2, sys.argv[2], w)
    w.close()
else:
    run_demo(generator.generator_coro)


gym.destroy_sim(sim)
