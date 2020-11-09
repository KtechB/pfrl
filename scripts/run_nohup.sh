SCRIPT="scripts/train_sac.sh"
LOGDIR="nohuplogs/sac"
mkdir $LOGDIR
ENVLIST=("HalfCheetah-v2" "Hopper-v2" "Ant-v2" "Walker2d-v2")
# ENVLIST=("HalfCheetah-v2" "Hopper-v2")
for ENV in ${ENVLIST[@]}
do
    echo $ENV
    nohup bash $SCRIPT $ENV > $LOGDIR/$ENV &
done