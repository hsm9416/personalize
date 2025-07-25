from xml.parsers.expat import model
import mujoco
import os
import numpy as np
import mujoco.viewer
import time
from model_loader import load_model
import model_loader
import math
from scipy.spatial.transform import Rotation as R


def join_configurations(model, data):

    model, data = model_loader.load_model("./models/humanoid.xml")

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


def quaternion_to_rotation_matrix(quat):

    q0, q1, q2, q3 = quat

    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    r10 = 2 * (q1 * q2 + q0 * q3) 
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
    return rot_matrix
    


def floating_base_jacobian(model, data):
    """
    MuJoCo에서 floating base의 Jacobian을 계산합니다.
    """
    # MuJoCo에서 Jacobian 계산
    mujoco.mj_jac(model, data, data.jacp, data.jacr, data.jacv, 0)

    quat = data.qpos[0:4]# Unit quaternion: [w, x, y, z]

    R_b = quaternion_to_rotation_matrix(quat)  # 쿼터니언을 회전 행렬로 변환
    

    # Jacobian을 numpy 배열로 변환
    jacobian = np.zeros((6, model.nv))
    jacobian[:3, :] = data.jacp.reshape(3, model.nv)
    jacobian[3:, :] = data.jacr.reshape(3, model.nv)

    I_3 = np.eye(3)
    j_xb_1 = np.hstack((I_3, R_b))
    j_xb_2 = np.hstack((np.zeros((3, 3)), I_3))
    j_xb_total = np.vstack((j_xb_1, j_xb_2))

    return j_xb_total

