import math

import PyKDL
import kdl_parser_py
import kdl_parser_py.urdf

def move_bot_to(bot, x, y, z, wait=False):
    chain = bot.chain
    fkpos = PyKDL.ChainFkSolverPos_recursive(chain)
    ikvel = PyKDL.ChainIkSolverVel_pinv(chain)
    ik = PyKDL.ChainIkSolverPos_NR(chain, fkpos, ikvel)

    q_init = PyKDL.JntArray(7)
    pos_old = bot.get_joint_pos()
    for i in range(7):
        q_init[i] = pos_old[i]

    p_in = PyKDL.Frame()
    p_in.p.x(x)
    p_in.p.y(y)
    p_in.p.z(z)
    p_in.M.DoRotY(math.pi)
    print(p_in)
    q_out = PyKDL.JntArray(7)

    if ik.CartToJnt(q_init, p_in, q_out) != 0:
        print('IK failed')
        return

    yield from bot.set_joint_target_pos(list(q_out), wait=wait)

def set_bot_velocity(chain, x, y, z):
    pass



def generator_coro(bot):
    x = 0
    '''
    yield from bot.set_joint_target_pos([0,-0.5,0,0,0,0,0])
    yield 5
    print('yield1')
    x = 6
    yield from bot.set_joint_target_pos([0,0.5,0,-0.5,0,-1,0], wait=True)
    print('yield2')
    yield 5
    '''
    yield from move_bot_to(bot, 0.4, 0.0, 0.8, True)
    yield from bot.point_gripper(True)
    yield from move_bot_to(bot, 0.4, 0.0, 0.35, True)
    yield from move_bot_to(bot, 0.6, -0.06, 0.35, True)
    #while True:
    #    yield

def generator_coro2(bot, r, s):
    yield from move_bot_to(bot, 0.4, 0.0, 0.8, True)
    yield from bot.point_gripper(True)
    yield from move_bot_to(bot, 0.4, 0.0, 0.35, True)
    #yield from move_bot_to(bot, 0.6+0.1*(2*r[0]-1), 0.1*(2*r[1]-1), 0.35, True)
    yield from move_bot_to(bot, 0.6+0.1*(2*s[0]-1), 0.1*(2*s[1]-1), 0.35, True)
    #while True:
    #    yield
