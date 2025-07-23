from xml.parsers.expat import model
import mujoco
import os
import numpy as np
import mujoco.viewer
import time
from model_loader import load_model
import model_loader


def join_positions_to_quaternion(joint_positions):
    
    model, data = model_loader.load_model("./models/humanoid.xml")
    data.qpos[:] = 0.0           # INITIALIZE