#!/bin/bash

##############################################
### EXAMPLE TO RUN MANY JOBS WITH things   ###
### CREATE THE things FILES INTO A LOOP    ###
##############################################
#
# Borrowed from Théo Desbordes, changes by Mathias Sablé-Meyer. Have fun.
#
# Arguments:
# $1: job names
# $2: file containing the python file to run
# $3: file containing the "condition matrix" to run on

# Warn and exit if there is not input file
if [ -z "$1" ]; then
  echo 'Expecting an job name'
  exit
fi

# Warn and exit if there is not input file
if [ -z "$2" ]; then
  echo 'Expecting an input file containing the scripe to run... exiting'
  exit
fi

# Warn and exit if there is not input file
if [ -z "$3" ]; then
  echo 'Expecting an input file containing the parameters to run... exiting'
  exit
fi

# load commands from file
IFS=$'\r\n' GLOBIGNORE='*' command eval  'job_array=($(cat $3))'

# set up paths
timestamp=$(date "+%Y-%m-%d_%Hh%M")
logs_path='./logs/'$timestamp
mkdir -p $logs_path

for i in `seq 0 $((${#job_array[@]} - 1))`;
do
  file_sbatch=$logs_path/$1_$i.sbatch
  out_file=$logs_path/$1_$i.out
  err_file=$logs_path/$1_$i.err


cat <<EOT >> $file_sbatch
#!/bin/bash
#SBATCH --job-name=${1}_${i}
#SBATCH --output=$out_file
#SBATCH --error=$err_file
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=0-01:00

source ~/.bashrc
micromamba activate spyder_env
python $2 ${job_array[i]}
EOT

sbatch $file_sbatch

done
