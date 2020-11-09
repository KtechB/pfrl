ENVLIST=("HalfCheetah-v2" "Hopper-v2" "Ant-v2" "Walker2d-v2")
for ENV in ${ENVLIST[@]}
do
    nohup scripts/run_sac $ENV &
done