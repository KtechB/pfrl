"""A training script of Soft Actor-Critic on OpenAI Gym Mujoco environments.

This script follows the settings of https://arxiv.org/abs/1812.05905 as much
as possible.
i"""
import os
import argparse
import functools
import logging
import sys
from distutils.version import LooseVersion

import gym
import gym.wrappers
import numpy as np
import torch
from torch import distributions, nn

import pfrl
from pfrl import experiments, replay_buffers, utils
from pfrl.nn.lmbda import Lambda
# from sample_demonstration.sample_demo import sample_demonstration

import numpy as np
import os
import pickle

def sample_one_epis(env, agent, max_episode_len=None):
    env.spec.timestep_limit = max_episode_len
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
            next_o, r, done, _ = env.step(a)
            R += r
            t += 1
            obs.append(o)
            acs.append(a)
            rews.append(r)
            dones.append(done)
            o = next_o
            
            reset = done  or t == max_episode_len+1 #or info.get("needs_reset", False)
            agent.observe(o, r, done, reset)
            if done or reset:
                break
    
    epi = dict(
        obs=np.array(obs, dtype='float32'),
        acs=np.array(acs, dtype='float32'),
        rews=np.array(rews, dtype='float32'),
        dones=np.array(dones, dtype='float32'),
    )
    return epi, t, R  

def sample_demonstration(env, agent,  n_episodes, outputdir, model_path, max_episode_len=None):
    demo_name = "demonstrations.pkl"
    epis = []
    epi_len_list =[]
    rew_sum_list = []
    os.makedirs(outputdir, exist_ok=True)
    
    for i in range(n_episodes):
        epi, epi_length, rew_sum = sample_one_epis(env, agent, max_episode_len)
        epis.append(epi)
        epi_len_list.append(epi_length)
        rew_sum_list.append(rew_sum)

    with open(os.path.join(outputdir,demo_name ), "wb") as f:
        pickle.dump(epis, f)
    log_message = outputdir.split("/")[-1] + "  rewards:" + str(rew_sum_list)  + "\n"+ " polpath:"+ model_path + "\n"
    with open(os.path.join(outputdir, "reward_logs.txt"), mode = "a") as f:
        f.write(log_message)
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/demonstrations",
        help=(
            "Directory path to save output files."
        ),
    )
    
    parser.add_argument(
        "--env",
        type=str,
        default="Hopper-v2",
        help="OpenAI Gym MuJoCo env to perform algorithm on.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of envs run in parallel."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10 ** 6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )

    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.WARNING, help="Level of the root logger."
    )

    parser.add_argument(
        "--policy-output-scale",
        type=float,
        default=1.0,
        help="Weight initialization scale of policy output.",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=10, help="Number of envs run in parallel."
    )
   
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)


    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    def make_env(process_idx, test):
        env = gym.make(args.env)
        # Unwrap TimiLimit wrapper
        assert isinstance(env, gym.wrappers.TimeLimit)
        # env = env.env
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        # Normalize action space to [-1, 1]^n
        env = pfrl.wrappers.NormalizeActionSpace(env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir+"/video",video_callable=(lambda ep: ep % 1 == 0),force=True)
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )

    sample_env = make_env(process_idx=0, test=False)
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    obs_size = obs_space.low.size
    action_size = action_space.low.size

    if LooseVersion(torch.__version__) < LooseVersion("1.5.0"):
        raise Exception("This script requires a PyTorch version >= 1.5.0")

    def squashed_diagonal_gaussian_head(x):
        assert x.shape[-1] == action_size * 2
        mean, log_scale = torch.chunk(x, 2, dim=1)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = distributions.Independent(
            distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        # cache_size=1 is required for numerical stability
        return distributions.transformed_distribution.TransformedDistribution(
            base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
        )

    policy = nn.Sequential(
        nn.Linear(obs_size, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, action_size * 2),
        Lambda(squashed_diagonal_gaussian_head),
    )
    torch.nn.init.xavier_uniform_(policy[0].weight)
    torch.nn.init.xavier_uniform_(policy[2].weight)
    torch.nn.init.xavier_uniform_(policy[4].weight, gain=args.policy_output_scale)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    def make_q_func_with_optimizer():
        q_func = nn.Sequential(
            pfrl.nn.ConcatObsAndAction(),
            nn.Linear(obs_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        torch.nn.init.xavier_uniform_(q_func[1].weight)
        torch.nn.init.xavier_uniform_(q_func[3].weight)
        torch.nn.init.xavier_uniform_(q_func[5].weight)
        q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=3e-4)
        return q_func, q_func_optimizer

    q_func1, q_func1_optimizer = make_q_func_with_optimizer()
    q_func2, q_func2_optimizer = make_q_func_with_optimizer()

    rbuf = replay_buffers.ReplayBuffer(10 ** 6)

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = pfrl.agents.SoftActorCritic(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=0.99,
        replay_start_size=10000,
        gpu=args.gpu,
        minibatch_size=256,
        burnin_action_func=burnin_action_func,
        entropy_target=-action_size,
        temperature_optimizer_lr=3e-4,
    )

    if len(args.load) > 0:
        agent.load(args.load)
    outdir = args.outdir
    print("save to :",outdir)

    sample_demonstration(sample_env, agent, args.n_episodes, outdir, model_path = args.load,max_episode_len=timestep_limit)
if __name__ == "__main__":
    main()
