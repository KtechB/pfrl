ENV_NAME=Hopper-v2 #HalfCheetahPyBulletEnv-v0  #Pendulum-v0
LOG_DIR=../garbage/mujoco/$ENV_NAME/sac
UNIT_SIZE=256

python sample_demonstration/sample_demo_with_sac.py \
--env "$ENV_NAME" \
--outdir "results/demonstration/${ENV_NAME}/sac" \
--load "results/sac/Hopper-v2/9be4726d327b7ce32d9008c40119c98c93febad5-7b2a99a4-eb6e5bdd/300000_checkpoint" \
--gpu 1 \
--n_episodes 10 \
--monitor \
