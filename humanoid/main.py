from xml.parsers.expat import model
import mujoco
import os
import numpy as np
import mujoco.viewer
import time
import inverse_kinematics
from model_loader import load_model

if __name__ == "__main__":
    model, data = load_model()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_wall_time = time.time()
        gravity_on = False

        while viewer.is_running():
            elapsed = time.time() - start_wall_time
            if not gravity_on and elapsed >= 1.0:
                model.opt.gravity[2] = -9.81
                gravity_on = True

            mujoco.mj_step(model, data)
            viewer.sync()

            nv = model.nv
            k = 6
            m = 3

            J_c = np.random.randn(k, nv)          # Constraint Jacobian (dummy)
            J = np.random.randn(m, nv)            # Task Jacobian for checking condition 2 (dummy)
            J_task = np.random.randn(m, nv)       # Task Jacobian (e.g. hand)
            Jc_Xb = J_c[:, -6:]                   # Last 6 columns as base-related constraints
            q_dot = np.zeros(nv)                  # Current joint velocity (static assumption)
            xdot_desired = np.array([0.0, 0.0, -0.1])  # Task: downward motion

            qdot_final = inverse_kinematics.compute_ik_with_conditions(
                J, J_c, Jc_Xb, q_dot, J_task, xdot_desired
            )

            print(f"▶ qdot: {qdot_final}")

            sim_dt = model.opt.timestep
            loop_duration = time.time() - start_wall_time
            sleep_time = sim_dt - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)

# if __name__ == "__main__":

#     model, data = load_model()

#     # q_total = join_configurations(model, data)

#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         start_wall_time = time.time()
#         gravity_on = False

#         while viewer.is_running():
#             # 실제 시간 기준 경과 시간 계산
#             elapsed = time.time() - start_wall_time

#             # 실제 5초 경과 후 중력 복원
#             if not gravity_on and elapsed >= 1.0: 
#                model.opt.gravity[2] = -9.81
#                gravity_on = True
#             #    print(f"▶ 중력 복원됨 (실시간 경과: {elapsed:.2f}s)")

#             mujoco.mj_step(model, data)
#             viewer.sync()

#             J_cog = floating_base_jacobians.compute_cog_jacobian(model, data)

#             nv = model.nv
#             k = 6    # number of constraints (e.g., foot contacts)
#             m = 3    # task dimension (e.g., end-effector xyz velocity)

#             J_c = np.random.randn(k, nv)     # constraint Jacobian
#             J_task = np.random.randn(m, nv)  # task Jacobian (e.g. hand or foot)
#             xdot_desired = np.array([0.0, 0.0, -0.1])  # move downward

#             qdot_2= inverse_kinematics.compute_floating_base_qdot(J_c, J_task, xdot_desired)

#             print(f"▶ qdot(CONDITION2): {qdot_2}")

#             # print(q_total.shape)  # Print the total configuration state

#             sim_dt = model.opt.timestep
#             loop_duration = time.time() - start_wall_time          
#             sleep_time = sim_dt - loop_duration
#             if sleep_time > 0:
#                 time.sleep(sleep_time)