#!/bin/sh
#SBATCH -N 1

MAX_WAIT_TIME_STUCK=1500

while true; do
    # these three lines just to get the space separated list of runing job ids 
    squeue --user 40017027 -t RUNNING --format=%i -h > tmp_squeue.txt
    running_job_id_list=`sed ':a;N;$!ba;s/\n/ /g' tmp_squeue.txt`
    rm -f tmp_squeue.txt



    job_ids_with_epoch_output=''
    for file in slurm*; do
        echo "checking file ${file}"
        # if epoch was not printed in client.out file, then job_id will be None.
        job_id=`python3 scripts/cluster_scripts/check_jobs_state.py ${file} job_id`
        job_ids_with_epoch_output="$job_ids_with_epoch_output $job_id"
    done


    for job_id in $running_job_id_list; do
        echo "checking job ${job_id}"

        job_has_epoch_output=`python3 -c "print('${job_id}' in '${job_ids_with_epoch_output}')"`

        # these two lines to get the runtime of job in seconds
        runtime_job=`squeue --user 40017027 --job 51603 --Format=timeused -h`
        runtime_job=$(echo $runtime_job | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')

        echo "$job_id has job_has_epoch_output= $job_has_epoch_output"

        # Cancel job if it produced no "epoch" output within $MAX_WAIT_TIME_STUCK seconds
        if [[ $job_has_epoch_output == "False" ]]; then
            if [ "$runtime_job" -gt "$MAX_WAIT_TIME_STUCK" ]; then
                echo "(1) cancel_job $job_id"
                scancel $job_id
            fi
        fi
    done


    # Cancel job if it produced no "epoch" output within $MAX_WAIT_TIME_STUCK seconds
    for file in slurm*; do
        echo "checking file ${file}"
        job_id=`python3 scripts/cluster_scripts/check_jobs_state.py ${file} job_id`
        job_is_running=`python3 -c "print('${job_id}' in '${running_job_id_list}')"`

        # # This can be useful to compute how long it usually takes to do an epoch()
        # time_last_two=`python3 scripts/cluster_scripts/check_jobs_state.py ${file} time_last_two`
        # echo $file " -> job_id ${job_id}, time between last two epochs ${time_last_two}"


        if [[ $job_is_running == "True" ]]; then
            time=`python3 scripts/cluster_scripts/check_jobs_state.py ${file} time`
            # Cancel job if more than $MAX_WAIT_TIME_STUCK seconds passed since last epoch
            if [ "$time" -gt "$MAX_WAIT_TIME_STUCK" ]; then
                echo "(2) cancel_job $job_id"
                scancel $job_id
            else
                echo "(2) do not cancel_job $job_id"
            fi

        fi

    done
    echo "done iteration"
    sleep 300

done