ENV_NAME=Ant-v2 #HalfCheetahPyBulletEnv-v0  #Pendulum-v0
LOG_DIR=../garbage/mujoco/$ENV_NAME/sac
UNIT_SIZE=256

python sample_demonstration/sample_demo_with_sac.py \
--env "$ENV_NAME" \
--outrootdir "results/demonstration/${ENV_NAME}/sac" \
--load "results/sac/Ant-v2/9be4726d327b7ce32d9008c40119c98c93febad5-7b2a99a4-458966e0/300000_checkpoint" \
--gpu 1 \
--n_episodes 10 \
--batch-size 256 \
--monitor \
