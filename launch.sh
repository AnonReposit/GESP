#!/bin/bash
set -e

EXPERIMENT=""
PORT=""
CLUSTER=false
PARALLEL=""
SIMULATOR=""
for i in "$@"
do
case $i in
    -b|--build)
    bash build.sh
    ;;
    -e=*|--experiment=*)
    EXPERIMENT="${i#*=}"
    ;;
    --cluster)
    CLUSTER=true
    ;;
    -p=*|--port=*)
    PORT="${i#*=}"
    ;;
    --sequential)
    PARALLEL=false
    ;;
    --parallel)
    PARALLEL=true
    ;;
    --vrep)
    echo "Launching with vrep..."
    SIMULATOR=vrep
    ;;
    --coppelia)
    echo "Launching with coppelia..."
    SIMULATOR=coppelia
    ;;

    --default)
    DEFAULT=YES
    ;;
    *)
            # unknown option
    ;;
esac
done


if [[ -z $EXPERIMENT || ( $EXPERIMENT != "nipes" && $EXPERIMENT != "mnipes" ) ]]; then
    echo "Parameter file for simulation not provided."
    echo "Example: "
    echo ""
    echo "bash launch.sh -e=nipes"
    echo ""
    echo "Parameters for launch.sh script: "
    echo "-e=experiment_name       should be either nipes or mnipes"
    echo "-b                       build before launch"
    echo "--cluster                required when launching from napier uni cluster"
    echo "--port                   required when launching from napier uni cluster"
    echo "Exiting..."
    exit 1
fi


  if [[ "$PARALLEL" == "" ]]; then
    echo "ERROR: use parameter --sequential or --parallel to specify 
    wether to execute sequentially or in parallel. Exiting..."
    exit 1
  fi


folder_in_which_launchsh_is=`pwd`
if [[ "$CLUSTER" == true ]]; then
  if [[ "$PORT" == "" ]]; then
    echo "ERROR: cluster mode requires port parameter. 
    Choose a multiple of 10000 for the port parameter. 
    Each port should only be used by each process once. Exiting..."
    exit 1
  fi
fi



unique_experiment_id="`date +%s`$RANDOM$RANDOM$RANDOM"
unique_experiment_name="$EXPERIMENT$unique_experiment_id"
if [[ "$CLUSTER" == true ]]; then
  experiment_folder="$folder_in_which_launchsh_is/../evolutionary_robotics_framework/experiments/$unique_experiment_name"
else
  experiment_folder="$folder_in_which_launchsh_is/evolutionary_robotics_framework/experiments/$unique_experiment_name"
fi






# logFileMountpoint="/home/paran/Dropbox/BCAM/07_estancia_1/code/logs/"
# if grep -qs "$logFileMountpoint " /proc/mounts; then
#     echo "log dir is mounted."
# else
#     echo "log dir is not mounted. Mounting..."
#     sudo mount -t tmpfs -o size=300m tmpfs $logFileMountpoint
#     echo "mounted!"
# fi

mkdir $experiment_folder
cp $folder_in_which_launchsh_is/experiments/$EXPERIMENT/parameters.csv $experiment_folder/parameters.csv  
python3 scripts/utils/UpdateParameter.py -f $experiment_folder/parameters.csv -n experimentName -v "$unique_experiment_name"


if [[ "$CLUSTER" == true ]]; then
  python3 scripts/utils/UpdateParameter.py -f $experiment_folder/parameters.csv -n expPluginName -v "/share/earza/library/lib/" --updateOnlyPath
  python3 scripts/utils/UpdateParameter.py -f $experiment_folder/parameters.csv -n scenePath -v "$folder_in_which_launchsh_is/../evolutionary_robotics_framework/simulation/models/scenes/" --updateOnlyPath
  python3 scripts/utils/UpdateParameter.py -f $experiment_folder/parameters.csv -n robotPath -v "$folder_in_which_launchsh_is/../evolutionary_robotics_framework/simulation/models/robots/" --updateOnlyPath
  python3 scripts/utils/UpdateParameter.py -f $experiment_folder/parameters.csv -n modelsPath -v "$folder_in_which_launchsh_is/../evolutionary_robotics_framework/simulation/models/"
  python3 scripts/utils/UpdateParameter.py -f $experiment_folder/parameters.csv -n repository -v "$folder_in_which_launchsh_is/logs"
  export LD_LIBRARY_PATH=/share/earza/library/lib/

  if [[ "$PARALLEL" == true ]]; then
    python3 scripts/utils/UpdateParameter.py -f $experiment_folder/parameters.csv -n instanceType -v 1
    sbatch --job-name=earza scripts/cluster_scripts/are-parallel.job $unique_experiment_name $PORT 32 $SIMULATOR
  else
    sbatch --job-name=earza scripts/cluster_scripts/are-sequential.job $unique_experiment_name $SIMULATOR
  fi

else

  # https://stackoverflow.com/questions/2129923/how-to-run-a-command-before-a-bash-script-exits
  function cleanup {
    echo "Removing parameter files."
    rm -f -r $experiment_folder
  }
  trap cleanup EXIT

  # in lcluster mode, we cannot clean the experiment folder here bc we are doing an sbatch.

  if [[ "$SIMULATOR" == "vrep" ]]; then
    echo "ERROR: VREP does not work on my laptop."
    echo "EXITTING..."
    exit 1
    export LD_LIBRARY_PATH=/home/paran/Dropbox/BCAM/07_estancia_1/code/V-REP_PRO_EDU_V3_6_2_Ubuntu18_04
    ./V-REP_PRO_EDU_V3_6_2_Ubuntu18_04/vrep.sh -h -g$experiment_folder/parameters.csv
  elif [[ "$SIMULATOR" == "coppelia" ]]; then

    if [[ "$PARALLEL" == true ]]; then
      python3 scripts/utils/UpdateParameter.py -f $experiment_folder/parameters.csv -n instanceType -v 1
      python3 evolutionary_robotics_framework/simulation/Cluster/run_cluster.py --xvfb 1 --params $experiment_folder/parameters.csv --client /usr/local/bin/are-client --vrep CoppeliaSim_Edu_V4_3_0_Ubuntu18_04/are_sim.sh --port-start 10000 4
    else
      rm ./evolutionary_robotics_framework/CoppeliaSim_Edu_V4_3_0_Ubuntu18_04/libsimExtGenerate.so -f
      export LD_LIBRARY_PATH=/home/paran/Dropbox/BCAM/07_estancia_1/code/evolutionary_robotics_framework/CoppeliaSim_Edu_V4_3_0_Ubuntu18_04
      ./evolutionary_robotics_framework/CoppeliaSim_Edu_V4_3_0_Ubuntu18_04/are_sim.sh simulation -g$experiment_folder/parameters.csv
    fi

  else
    echo "ERROR: SIMULATOR must be either vrep or coppelia. SIMULATOR=$SIMULATOR was given."
  fi
fi

# Launch local in parallel, for debugging purposes
# cp experiments/nipes/parameters.csv evolutionary_robotics_framework/experiments/nipes/parameters.csv &&  python3 evolutionary_robotics_framework/simulation/Cluster/run_cluster.py --xvfb 1 --params /home/paran/Dropbox/BCAM/07_estancia_1/code/experiments/nipes/parameters.csv --client /usr/local/bin/are-client --vrep CoppeliaSim_Edu_V4_3_0_Ubuntu18_04/are_sim.sh --port-start 10000 4