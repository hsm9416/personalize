import floating_base_jacobians
import mujoco
import numpy as np
import function



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

    # Stack constraints + task
    J_stack = np.vstack([J_c, J_task])        # Shape: (k+m, nv)
    x_stack = np.concatenate([np.zeros(J_c.shape[0]), xdot_desired])  # Shape: (k+m,)

    # Compute pseudo-inverse via damped least-squares or standard method
    JT = J_stack.T
    JJT_inv = np.linalg.pinv(J_stack @ JT)    # right pseudo-inverse

    qdot = JT @ JJT_inv @ x_stack             # Eq. (8)
    return qdot
