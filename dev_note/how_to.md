
# quickstart
Q-functionなどはtorch.nn.Moduleからの継承で作れる
forwardのoutputはpfrl.action_value.DecreteActionValue()などでWrapするう

エージェントの作成
```
 Set the discount factor that discounts future rewards.
gamma = 0.9

# Use epsilon-greedy for exploration
explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# As PyTorch only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(numpy.float32, copy=False)

# Set the device id to use GPU. To use CPU only, set it to -1.
gpu = -1

# Now create an agent that will interact with the environment.
agent = pfrl.agents.DoubleDQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma,
    explorer,
    replay_start_size=500,
    update_interval=1,
    target_update_interval=100,
    phi=phi,
    gpu=gpu,
)
```

agentインスタンスでデータを管理するイメージ　
行動サンプル及び評価はべた書き

experiments.train_agent_with_evaluationで省略も可能
<pre>
.
├── __init__.py
├── action_value.py
├── agent.py
├── agents : 各種手法
│   ├── __init__.py
│   ├── a2c.py ......

├── collections : データ構造
│   ├── __init__.py
│   ├── persistent_collections.py
│   ├── prioritized.py
│   └── random_access_queue.py
├── distributions　：デルタ分布
│   ├── __init__.py
│   └── delta.py
├── env.py
├── envs
│   ├── __init__.py
│   ├── abc.py　：テスト用の環境
│   ├── multiprocess_vector_env.py
│   └── serial_vector_env.py
├── experiments　：実験省略用
│   ├── __init__.py
│   ├── evaluator.py
│   ├── hooks.py
│   ├── prepare_output_dir.py
│   ├── train_agent.py
│   ├── train_agent_async.py
│   └── train_agent_batch.py
├── explorer.py
├── explorers　：要はアクター
│   ├── __init__.py
│   ├── additive_gaussian.py
│   ├── additive_ou.py
│   ├── boltzmann.py
│   ├── epsilon_greedy.py
│   └── greedy.py
├── functions
│   ├── __init__.py
│   ├── bound_by_tanh.py
│   └── lower_triangular_matrix.py
├── initializers
│   ├── __init__.py
│   ├── chainer_default.py
│   └── lecun_normal.py
├── nn　：ネットワーク系統
│   ├── __init__.py
│   ├── atari_cnn.py
│   ├── bound_by_tanh.py
│   ├── branched.py
│   ├── concat_obs_and_action.py
│   ├── empirical_normalization.py
│   ├── lmbda.py
│   ├── mlp.py
│   ├── mlp_bn.py
│   ├── noisy_chain.py
│   ├── noisy_linear.py
│   ├── recurrent.py
│   ├── recurrent_branched.py
│   └── recurrent_sequential.py
├── optimizers
│   ├── __init__.py
│   └── rmsprop_eps_inside_sqrt.py
├── policies
│   ├── __init__.py
│   ├── deterministic_policy.py
│   ├── gaussian_policy.py
│   └── softmax_policy.py
├── policy.py
├── q_function.py
├── q_functions
│   ├── __init__.py
│   ├── dueling_dqn.py
│   ├── state_action_q_functions.py
│   └── state_q_functions.py
├── replay_buffer.py
├── replay_buffers
│   ├── __init__.py
│   ├── episodic.py
│   ├── persistent.py
│   ├── prioritized.py
│   ├── prioritized_episodic.py
│   └── replay_buffer.py
├── testing.py
├── utils
│   ├── __init__.py
│   ├── ask_yes_no.py
│   ├── async_.py
│   ├── batch_states.py
│   ├── clip_l2_grad_norm.py
│   ├── conjugate_gradient.py
│   ├── contexts.py
│   ├── copy_param.py
│   ├── env_modifiers.py
│   ├── is_return_code_zero.py
│   ├── mode_of_distribution.py
│   ├── pretrained_models.py
│   ├── random.py
│   ├── random_seed.py
│   ├── recurrent.py
│   ├── reward_filter.py
│   └── stoppable_thread.py
└── wrappers
    ├── __init__.py
    ├── atari_wrappers.py
    ├── cast_observation.py
    ├── continuing_time_limit.py
    ├── monitor.py
    ├── normalize_action_space.py
    ├── randomize_action.py
    ├── render.py
    ├── scale_reward.py
    └── vector_frame_stack.py

</pre>

## 基本的な使い方
Agentを作る
- policy-value networkを作成
- optimizer
- replaybufferを作成
- 初期状態での行動方策を作成

train_agent_batch_with_evaluation()を実行
```
nohu


## modelの保存場所
savedirectoryのbestに.pt拡張子で各attribute(アルゴリズムによってそれぞれ)が保存される形式