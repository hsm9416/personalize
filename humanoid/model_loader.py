from xml.parsers.expat import model
import mujoco
import os
import numpy as np
import mujoco.viewer
import time


def load_model(model_path):
    model_path = os.path.join(os.path.dirname(__file__), "./models/humanoid.xml")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Cannot find XML at {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)    


    data.qpos[:] = 0.0           # INITIALIZE
    data.qpos[0:4] = [0, 0, 5, 0] # free-joint quaternion
    data.qpos[4:7] = [0, 0, 1]  # base 위치 (x,y,z)
    data.qpos[7:] = 0.0
    data.qvel[:] = 0.0                                                                                                                                                                                                                                                    
    data.ctrl[:] = 0.0

    model.opt.gravity[:] = 0
    mujoco.mj_step(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_wall_time = time.time()
        gravity_on = False

        while viewer.is_running():
            # 실제 시간 기준 경과 시간 계산
            elapsed = time.time() - start_wall_time

            # 실제 5초 경과 후 중력 복원
            if not gravity_on and elapsed >= 3.0:
               model.opt.gravity[2] = -9.81
               gravity_on = True
            #    print(f"▶ 중력 복원됨 (실시간 경과: {elapsed:.2f}s)")

            mujoco.mj_step(model, data)
            viewer.sync()
            
            sim_dt = model.opt.timestep
            loop_duration = time.time() - start_wall_time          
            sleep_time = sim_dt - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)

    return model, data
