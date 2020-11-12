ENV_NAME=Walker2d-v2 
LOG_DIR=../garbage/mujoco/$ENV_NAME/sac
UNIT_SIZE=256

python sample_demonstration/sample_demo_with_sac.py \
--env "$ENV_NAME" \
--outdir "results/demonstration/${ENV_NAME}/sac" \
--load "/home/hirobuchi.ryota/RLs/pfrl/results/sac/Walker2d-v2/9be4726d327b7ce32d9008c40119c98c93febad5-7b2a99a4-66cf37b0/300000_checkpoint" \
--gpu 1 \
--n_episodes 10 \
--monitor \
