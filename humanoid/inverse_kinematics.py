import floating_base_jacobians
import mujoco
import numpy as np
import function


def check_condition_1(Jc, q_dot, tol=1e-6):
    Xc_dot = Jc @ q_dot
    return np.all(np.abs(Xc_dot) < tol)

def check_condition_2(J, Jc):
    stacked = np.vstack((J, Jc))
    return np.linalg.matrix_rank(stacked) == stacked.shape[0]

def check_condition_3(Jc_Xb):
    return np.linalg.matrix_rank(Jc_Xb) == 6

def compute_floating_base_qdot(J_c, J_task, xdot_desired):
    """
    Computes qdot for floating-base system using orthogonal decomposition.
    
    Parameters:
        J_c      : (k x nv) constraint Jacobian
        J_task   : (m x nv) task-space Jacobian
        xdot_desired : (m,) desired end-effector velocity
    
    Returns:
        qdot : (nv,) joint velocity command
    """
    J_stack = np.vstack([J_c, J_task])
    x_stack = np.concatenate([np.zeros(J_c.shape[0]), xdot_desired])
    JT = J_stack.T
    JJT_inv = np.linalg.pinv(J_stack @ JT)
    qdot = JT @ JJT_inv @ x_stack
    return qdot

def compute_ik_with_conditions(J, J_c, Jc_Xb, q_dot, J_task, xdot_desired):
    if check_condition_1(J_c, q_dot):
        # print("▶ Using IK (Condition 1: Xc = 0)")
        return compute_floating_base_qdot(J_c, J_task, xdot_desired)
    elif check_condition_2(J, J_c):
        print("▶ Using IK (Condition 2: full rank)")
        return compute_floating_base_qdot(J_c, J_task, xdot_desired)
    elif check_condition_3(Jc_Xb):
        print("▶ Using IK (Condition 3: rank(Jc_Xb) == 6)")
        return compute_floating_base_qdot(J_c, J_task, xdot_desired)
    else:
        print("▶ No valid condition met! Returning zero qdot.")
        return np.zeros(J_c.shape[1])
