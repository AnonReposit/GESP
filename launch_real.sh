#!/bin/bash
set -e

SEED=""
SKIPINIT=false
for i in "$@"
do
case $i in
    --seed=*)
    SEED="${i#*=}"
    ;;
    --default)
    DEFAULT=YES
    ;;
    --skipinit)
    echo "Skip initialization"
    SKIPINIT=true
    ;;
    *)
            # unknown option
    ;;
esac
done


if [[ -z $SEED ]]; then
    echo "Argument --seed=i is required, where i is an integer."
    exit 1
fi

logFolder="logs/real_world_exp_${SEED}"
mkdir -p $logFolder

nEvals=`ls logs/real_world_exp_2/learner_* | wc -l`


if (( nEvals < 1 )); then
    # loadExistingControllers always needs to be zero
    python3 scripts/utils/UpdateParameter.py -f evolutionary_robotics_framework/experiments/physical_melai/parameters.csv -n loadExistingControllers -v "0"
    python3 scripts/utils/UpdateParameter.py -f evolutionary_robotics_framework/experiments/physical_melai/parameters.csv -n learnerToLoad -v " "
    cp -r logs/real_world_exp_template/* $logFolder/
else
    python3 scripts/utils/UpdateParameter.py -f evolutionary_robotics_framework/experiments/physical_melai/parameters.csv -n loadExistingControllers -v "1"
    python3 scripts/utils/UpdateParameter.py -f evolutionary_robotics_framework/experiments/physical_melai/parameters.csv -n learnerToLoad -v "`ls logs/real_world_exp_2/learner_* | xargs python3 -c "import sys;a = sorted(sys.argv[1:], key= lambda x: int(x.split('_')[-1])) ; print(a[-1])" | xargs basename`"
fi

python3 scripts/utils/UpdateParameter.py -f evolutionary_robotics_framework/experiments/physical_melai/parameters.csv -n experimentName -v real_world_exp_${SEED}
python3 scripts/utils/UpdateParameter.py -f evolutionary_robotics_framework/experiments/physical_melai/parameters.csv -n seed -v ${SEED}
python3 scripts/utils/UpdateParameter.py -f evolutionary_robotics_framework/experiments/physical_melai/parameters.csv -n resultFile -v /home/paran/Dropbox/BCAM/07_estancia_1/code/results/physical_data/result_constant_${SEED}.txt


if [ "$SKIPINIT" = false ]; then
    gnome-terminal -x sh -c "bash launch_tracking_system.sh; bash"
    gnome-terminal -x sh -c "bash connect_to_robot.sh; bash"


    echo "Waiting 15 seconds for the connections to stablish..."
    sleep 15
fi


are-update /home/paran/Dropbox/BCAM/07_estancia_1/code/evolutionary_robotics_framework/experiments/physical_melai/parameters.csv
