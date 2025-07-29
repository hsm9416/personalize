from xml.parsers.expat import model
import mujoco
import os
import numpy as np
import mujoco.viewer
import time
from model_loader import load_model
import model_loader
import math
import function
from scipy.spatial.transform import Rotation as R


def join_configurations(model, data):

    model, data = model_loader.load_model()

    # 초기화
    data.qpos[:] = 0.0
    data.qpos[0:4] = [1, 0, 0, 0]  # Unit quaternion: [w, x, y, z]
    data.qpos[4:7] = [0.0, 0.0, 1.0]  # Base 위치

    q_r = data.qpos[7:]  # JOINT POSITIONS
    # [rhip, rknee, rankle, lhip,lknee, lankle]
    

    # quaternion → euler 변환 (scipy는 [x, y, z, w] 순서 필요)
    quat_mj = data.qpos[0:4]  # MuJoCo 쿼터니언: [w, x, y, z]
    quat_scipy = [quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]]  # → [x, y, z, w]

    r = R.from_quat(quat_scipy)
    x_b_orientation = r.as_euler('xyz')  # roll, pitch, yaw
    x_b_pos = data.qpos[4:7]

    x_b = np.concatenate((x_b_pos, x_b_orientation))  # BASE STATE

    # 최종 q_total
    q_total_transpose = np.concatenate((x_b, q_r))
    q_total = np.transpose(q_total_transpose)

    print(f"▶ q_total: {q_total}")

    return q_total


def floating_base_jacobian(model, data):
    """
    MuJoCo에서 floating base의 Jacobian을 계산합니다.
    """
    # MuJoCo에서 Jacobian 계산
    mujoco.mj_jac(model, data, data.jacp, data.jacr, data.jacv, 0)

    quat = data.qpos[0:4]# Unit quaternion: [w, x, y, z]

    R_b = function.quaternion_to_rotation_matrix(quat)  # 쿼터니언을 회전 행렬로 변환
    
    n = model.njnt  # Number of joints

    # Jacobian을 numpy 배열로 변환
    jacobian = np.zeros((6, model.nv))
    jacobian[:3, :] = data.jacp.reshape(3, model.nv)
    jacobian[3:, :] = data.jacr.reshape(3, model.nv)

    I_3 = np.eye(3)
    J_xb_1 = np.hstack((I_3, R_b))
    J_xb_2 = np.hstack((np.zeros((3, 3)), I_3))
    J_xb_total = np.vstack((J_xb_1, J_xb_2))

    return J_xb_total


def compute_base_jacobian(model):
    n = model.nv  # Number of DOFs
    J_b = np.block([
        [np.zeros((6, n)), np.eye(6)]
    ])
    return J_b


def compute_cog_jacobian(model, data):
    """
    Compute Jacobian of the robot's center of gravity (CoG)
    
    Parameters:
        model: mujoco.MjModel
        data: mujoco.MjData
    Returns:
        J_cog: (3 x (nv+6)) full Jacobian matrix of the center of gravity
    """
    # List of bodies contributing to CoG
    link_bodies = [
        "torso", "right_hip", "right_knee", "right_ankle",
        "left_hip", "left_knee", "left_ankle"
    ]

    # Masses    
    m_list = [
    model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]
    for name in link_bodies

    ]
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    # Base frame
    x_b = data.xpos[torso_id]
    R_b = data.xmat[torso_id].reshape(3, 3)
    x_cog = data.subtree_com[torso_id]

    # Jacobians of each link's CoG
    J_list = []
    for name in link_bodies:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, os.name)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
        J_list.append(jacp)
        
    total_mass = sum(m_list)

    # Weighted average of CoG Jacobians
    J_cog_partial = sum(m * J for m, J in zip(m_list, J_list)) / total_mass

    # Full Jacobian [J_cog_partial | I | R_b × (x_cog - x_b)]
    I3 = np.eye(3)
    cross_term = R_b @ function.skew(x_cog - x_b)
    J_cog = np.hstack([J_cog_partial, I3, cross_term])

    return J_cog