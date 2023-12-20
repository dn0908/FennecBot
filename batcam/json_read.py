import glob
import os
import json
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
# read json
folder_path = '/home/smi/FennecBotData/'
file_path = glob.glob(f'{folder_path}/*.json')
file_path = max(file_path, key= os.path.getmtime) # get the latest json file
print("✅ Loading Data..... Reading file", file_path)

with open(file_path, "r") as f:
    data = json.load(f)

    # convert data to numpy
    x = np.array([d["Lmap_x"] for d in data])
    y = np.array([d["Lmap_y"] for d in data])
    p = np.array([d["probability"] for d in data])

    # Interpolation
    x_new = np.linspace(1, 40, 40)
    y_new = np.linspace(1, 30, 30)
    f = interpolate.interp2d(x, y, p, kind="linear")
    p_new = f(x_new, y_new)

    def on_key_press(event):
        if event.key == "q":
            filename = file_path.replace(".", "_")
            filename = filename + '.png'
            # plt.savefig(filename)
            mpimg.imsave(filename, p_new, cmap='hsv')
            plt.close()

    plt.imshow(p_new, cmap="hsv")
    plt.connect("key_press_event", on_key_press)
    plt.show()