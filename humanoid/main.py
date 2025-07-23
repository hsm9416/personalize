from xml.parsers.expat import model
import mujoco
import os
import numpy as np
import mujoco.viewer
import time
from model_loader import load_model



if __name__ == "__main__":
    load_model("./models/humanoid.xml")
