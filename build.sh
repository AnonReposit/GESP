#!/bin/bash

WORKING_DIR=`pwd`

set -e
BUILD_MODE=Release
CLUSTER=""
SIMULATOR=""
CLEAN=false
for i in "$@"
do
case $i in
    --clean)
    CLEAN=true
    ;;
    --debug)
    echo "Building in Debug mode..."
    BUILD_MODE=Debug
    ;;
    --cluster)
    echo "Building in Cluster..."
    CLUSTER=true
    evolutionary_robotics_framework="$WORKING_DIR/../evolutionary_robotics_framework"
    ;;
    --local)
    echo "Building locally..."
    CLUSTER=false
    evolutionary_robotics_framework="$WORKING_DIR/evolutionary_robotics_framework"
    ;;
    --vrep)
    echo "Building with vrep..."
    SIMULATOR=vrep
    ;;
    --coppelia)
    echo "Building with coppelia..."
    SIMULATOR=coppelia
    ;;
    *)
            # unknown option
    ;;
esac
done

if [[ "$CLEAN" == true ]]; then
    echo "Cleaning..."
    cd $evolutionary_robotics_framework/build
    make clean
    cd ..
    rm build -rf
    echo "cleaned."
    cd $WORKING_DIR
fi


if [ -z "$CLUSTER" ]; then 
    echo "ERROR: Use of either --cluster or --local parameter is required. Exit...";
    exit 1 
fi

if [ -z "$SIMULATOR" ]; then 
    echo "ERROR: Use of either --vrep or --coppelia parameter is required. Exit...";
    exit 1 
fi

rm -rf "$evolutionary_robotics_framework/experiments/mnipes"
rm -rf "$evolutionary_robotics_framework/experiments/nipes"

rsync -r -v --exclude=*.csv "experiments/" "$evolutionary_robotics_framework/experiments/"

# BUILD_MODE=Debug
# BUILD_MODE=Release

cd $evolutionary_robotics_framework 
mkdir -p build
cd build


# Cluster.
if [[ "$CLUSTER" == true ]]; then

    if [[ "$SIMULATOR" == "vrep" ]]; then
        export LD_LIBRARY_PATH=/share/earza/library/lib/
        /share/earza/cmake/bin/cmake -DONLY_SIMULATION=1 -DCMAKE_INSTALL_PREFIX=/share/earza/library/ -DCOPPELIASIM_FOLDER=  -DVREP_FOLDER=$evolutionary_robotics_framework/../V-REP_PRO_EDU_V3_6_2_Ubuntu18_04 -DLIMBO_FOLDER=$evolutionary_robotics_framework/modules/are_limbo -DWITH_NN2=1 -DCMAKE_BUILD_TYPE=$BUILD_MODE ..
        make
        make install
    fi

    if [[ "$SIMULATOR" == "coppelia" ]]; then
        export LD_LIBRARY_PATH=/share/earza/library/lib/
        /share/earza/cmake/bin/cmake -DONLY_SIMULATION=1 -DCMAKE_INSTALL_PREFIX=/share/earza/library/ -DCOPPELIASIM_FOLDER=/share/earza/CoppeliaSim_Edu_V4_3_0_Ubuntu18_04  -DLIMBO_FOLDER=$evolutionary_robotics_framework/modules/are_limbo -DWITH_NN2=1 -DCMAKE_BUILD_TYPE=$BUILD_MODE ..
        make
        make install
    fi



# Local laptop.
else

    if [[ "$SIMULATOR" == "vrep" ]]; then
        cmake -DONLY_SIMULATION=0 -DCOPPELIASIM_FOLDER= -DVREP_FOLDER=$evolutionary_robotics_framework/../V-REP_PRO_EDU_V3_6_2_Ubuntu18_04 -DCMAKE_INSTALL_PREFIX=/usr/local/  -DLIMBO_FOLDER=$evolutionary_robotics_framework/modules/limbo -DWITH_NN2=1 -DCMAKE_BUILD_TYPE=$BUILD_MODE ..
    fi

    if [[ "$SIMULATOR" == "coppelia" ]]; then
        cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ -DONLY_SIMULATION=0 -DCOPPELIASIM_FOLDER=$evolutionary_robotics_framework/CoppeliaSim_Edu_V4_3_0_Ubuntu18_04 -DLIMBO_FOLDER=$evolutionary_robotics_framework/modules/limbo -DWITH_NN2=1 -DCMAKE_BUILD_TYPE=$BUILD_MODE ..
    fi

    make -j
    sudo make install
fi

echo "done compiling!"
cd $WORKING_DIR
