from model_loader import load_model
import floating_base_jacobians
import inverse_kinematics
import numpy as np
import mujoco
import time


if __name__ == "__main__":
    model, data = load_model()

    # 중력 끄기
    model.opt.gravity[:] = [0.0, 0.0, 0.0]

    # 공중에 초기 자세 설정
    data.qpos[:] = np.zeros_like(data.qpos)
    data.qpos[2] = 1.5  # 공중에 띄우기
    data.qpos[3:7] = [1.0, 0, 0, 0]

    

    with mujoco.viewer.launch_passive(model, data) as viewer:
        nv = model.nv

        left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle")
        right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle")
        print(f"left_foot_id: {left_foot_id}, right_foot_id: {right_foot_id}")

        left_foot_target = data.xpos[left_foot_id].copy() + np.array([0.1, 0.1, 0.1])
        right_foot_target = data.xpos[right_foot_id].copy() + np.array([-0.1, -0.1, 0.1])
        cog_xy_target = data.subtree_com[0][:2].copy()

        Kp = 1.0

        while viewer.is_running():
            mujoco.mj_step1(model, data)

            # No constraints!
            J_c = np.zeros((0, nv))
            Jc_Xb = np.zeros((0, 6))

            jacp_l = np.zeros((3, nv))
            jacr_l = np.zeros((3, nv))
            mujoco.mj_jacBody(model, data, jacp_l, jacr_l, left_foot_id)

            jacp_r = np.zeros((3, nv))
            jacr_r = np.zeros((3, nv))
            mujoco.mj_jacBody(model, data, jacp_r, jacr_r, right_foot_id)

            J_cog = floating_base_jacobians.compute_cog_jacobian(model, data)
            J_cog_xy = J_cog[:2, :nv]

            J_task = np.vstack([jacp_l, jacr_l, jacp_r, jacr_r, J_cog_xy])

            xpos_L = data.xpos[left_foot_id]
            xpos_R = data.xpos[right_foot_id]
            cog_xy = data.subtree_com[0][:2]


            xdot_desired = np.concatenate([
                Kp * (left_foot_target - xpos_L),
                np.zeros(3),
                Kp * (right_foot_target - xpos_R),
                np.zeros(3),
                Kp * (cog_xy_target - cog_xy)
            ])

            qdot_final = inverse_kinematics.compute_ik_with_conditions(
                J_task, J_c, Jc_Xb, data.qvel, J_task, xdot_desired
            )
            # print("left_foot_id:", left_foot_id, "right_foot_id:", right_foot_id)
            # print("xdot_desired:", xdot_desired.round(4))
            # print("qdot_final:", qdot_final.round(4))
            # print("qpos:", data.qpos.round(4))  

            data.qvel[:] = qdot_final

            mujoco.mj_step2(model, data)
            viewer.sync()