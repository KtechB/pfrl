import numpy as np
import os
import pickle

def sample_one_epis(env, agent, max_episode_len=None):
    with agent.eval_mode():
        obs = []
        acs =[]
        rews = []
        dones = []
        

        epis_num =0
            
        o = env.reset()
        R = 0
        t = 0
        epis_num +=1
        while True:
            a = agent.act(o)
            o, r, done, _ = env.step(a)
            R += r
            t += 1
            
            reset = done or t == max_episode_len+1 #or info.get("needs_reset", False)
            agent.observe(obs, r, done, reset)
            if done or reset:
                break
    
    epi = dict(
        obs=np.array(obs, dtype='float32'),
        acs=np.array(acs, dtype='float32'),
        rews=np.array(rews, dtype='float32'),
        dones=np.array(dones, dtype='float32'),
    )
    return epi, t, R  

def sample_demonstation(env, agent,  n_episodes, outputdir, model_path, max_episode_len=None):
    demo_name = "demonstrations.pkl"
    epis = []
    epi_len_list =[]
    rew_sum_list = []
    os.makedirs(outputdir, exist_ok=True)
    
    for i in n_episodes:
        epi, epi_length, rew_sum = sample_one_epis(env, agent, max_episode_len)
        epis.append(epi)
        epi_len_list.append(epi_length)
        rew_sum_list.append(rew_sum)

    with open(os.path.join(outputdir,demo_name ), "wb") as f:
        pickle.dump(epis, f)
    log_message = outputdir.split("/")[-1] + "  rewards:" + str(rew_sum_list)  + "\n"+ " polpath:"+ model_path + "\n"
    with open(os.path.join(outputdir, "reward_logs.txt"), mode = "a") as f:
        f.write(log_message)

    
