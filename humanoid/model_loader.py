from xml.parsers.expat import model
import mujoco
import os
import numpy as np
import mujoco.viewer
import time


def load_model(model_path):
    model_path = os.path.join(os.path.dirname(__file__), "./models/exo.xml")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Cannot find XML at {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)    


    data.qpos[:] = 0.0           # INITIALIZE
    data.qpos[0:4] = [0, 0, 0, 0] # free-joint quaternion
    data.qpos[4:7] = [0, 0, 1]  # base 위치 (x,y,z)
    data.qpos[7:] = 0.0
    data.qvel[:] = 0.0                                                                                                                                                                                                                                                    
    data.ctrl[:] = 0.0

    model.opt.gravity[:] = 0
    return model, data


    
