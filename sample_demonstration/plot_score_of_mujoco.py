
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

RESULT_DIR = "results/sac"
out_dir = "./results/graphs"
os.makedirs(out_dir, exist_ok=True)
env_list = ["HalfCheetah-v2", "Hopper-v2","Walker2d-v2","Ant-v2" ]
env_num = len(env_list)
plot_raw = math.floor(env_num/2)

fig , ax= plt.subplots( plot_raw, 2)

for i, env in enumerate(env_list):
    score_path = glob.glob(RESULT_DIR+"/"+env+  "/*/scores.txt")[0]
    print(score_path)
    

    scores = pd.read_csv(score_path, delimiter="\t")
    ax[ i//2,i%2].plot(scores["steps"], scores["mean"])
    ax[ i//2,i%2].set_title(env)
plt.savefig(os.path.join(out_dir, "mujoco_score.pdf"))