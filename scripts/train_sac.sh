ENV_NAME=$1 #HalfCheetahPyBulletEnv-v0  #Pendulum-v0
LOG_DIR=../garbage/mujoco/$ENV_NAME/sac
UNIT_SIZE=256

python examples/mujoco/reproduction/soft_actor_critic/train_soft_actor_critic.py \
--env "$ENV_NAME" \
--outdir "results/sac/${ENV_NAME}" \
--gpu 1 \
--steps 1000000 \
--eval-n-runs 5 \
--eval-interval 5000 \
--replay-start-size 10000 \
--batch-size 256 \
--log-interval 5000 \
--checkpoint_freq 100000 \
--log-level 3 \