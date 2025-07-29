from xml.parsers.expat import model
import mujoco
import os
import numpy as np
import mujoco.viewer
import time
import floating_base_jacobians
import inverse_kinematics
from model_loader import load_model


if __name__ == "__main__":

    model, data = load_model()

    # q_total = join_configurations(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_wall_time = time.time()
        gravity_on = False

        while viewer.is_running():
            # 실제 시간 기준 경과 시간 계산
            elapsed = time.time() - start_wall_time

            # 실제 5초 경과 후 중력 복원
            if not gravity_on and elapsed >= 1.0: 
               model.opt.gravity[2] = -9.81
               gravity_on = True
            #    print(f"▶ 중력 복원됨 (실시간 경과: {elapsed:.2f}s)")

            mujoco.mj_step(model, data)
            viewer.sync()

            J_cog = floating_base_jacobians.compute_cog_jacobian(model, data)

            nv = model.nv
            k = 6    # number of constraints (e.g., foot contacts)
            m = 3    # task dimension (e.g., end-effector xyz velocity)

            J_c = np.random.randn(k, nv)     # constraint Jacobian
            J_task = np.random.randn(m, nv)  # task Jacobian (e.g. hand or foot)
            xdot_desired = np.array([0.0, 0.0, -0.1])  # move downward

            qdot = inverse_kinematics.compute_floating_base_qdot(J_c, J_task, xdot_desired)


            print(f"▶ J_cog: {qdot}")

            # print(q_total.shape)  # Print the total configuration state

            sim_dt = model.opt.timestep
            loop_duration = time.time() - start_wall_time          
            sleep_time = sim_dt - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)