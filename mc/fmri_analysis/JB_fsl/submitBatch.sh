#!/bin/sh
# Create batches of jobs for parallel execution, then submit batches serially
# Usage example: ./submitBatch.sh dummy.sh 3 long.q s1 s2 s3 s4 s5 s6
# Since this script submits to jalapeno using fsl_sub, it needs to be executed on the server

# Command line argument 1/4: which script to execute
script=$1
shift

# Command line argument 2/4: batch size
batchSize=$1
shift

# Command line argument 3/4: which queue to submit to
queue=$1
shift

# Command line argument 4/4: arguments for each script (this sets how many times the script is executed in total)
arguments=$@

# Display input
echo This will generate batches of size $batchSize to execute ./$script with arguments $arguments

# Set analysis directory for execution on server
analysisDir=/home/fs0/jacobb/Analysis

# Set current batch id, counter to keep track of which batch is being filled
batchID=0

# Set number of script in current batch, counter to keep track of command in current batch
scriptID=0

# Get scriptname for use in filenames
scriptName=${script%%.*}

# Create directory for batch files
batchDir=submitBatch_${scriptName}
# Add a number to it so you don't replace existing directories of batch files
batchDirNum=0
# Find the first number for which the directory doesn't exist
while [ -d $analysisDir/${batchDir}_$(printf "%02d" $batchDirNum) ]; do
	echo ${batchDir}_$(printf "%02d" $batchDirNum) already exists
    batchDirNum=$((batchDirNum+1))
done
# And set the directory name for the found number
batchDir=${batchDir}_$(printf "%02d" $batchDirNum)
# Then make the directory
mkdir -p $analysisDir/$batchDir

# Run through arguments to generate file for current batch
for currArgument in $arguments ; do
	# If this is the first command: create file and add the first line
	if [ $scriptID = 0 ] ; then
		# Set filename for this batch
		batchFile=$analysisDir/$batchDir/${scriptName}_batch_$(printf "%02d" $batchDirNum)_$(printf "%02d" $batchID).sh
		# Write the shebang - it's nice but not necessary. Leave so it doesn't count as additional job
		# echo "#!/bin/sh" > $batchFile
		# Output status
		echo Created new file $batchFile for batch $batchID
	fi
	
	# Add current line to this batch file
	echo "$analysisDir/$script $currArgument" >> $batchFile
	
	# If this is the last command: set permissions and go to next batch
	if [ $scriptID = $((batchSize-1)) ] ; then
		# Set permissions
		chmod a+x $batchFile
		# Increase current batch
		batchID=$((batchID+1))
		# Reset script ID
		scriptID=0
	else
		# Increase current script ID
		scriptID=$((scriptID+1))
	fi
done

# If the last batch wasn't completely full: permission has not been updated yet
if [ $scriptID != 0 ] ; then
	# Set permissions
	chmod a+x $batchFile
fi

# Remember the id from the previously submitted job
previousJob=0;

# Finally: run batch files sequentially
for currBatch in $analysisDir/$batchDir/* ; do
	# The first job can be released directly, others have to wait until previous one finishes
	if [ $previousJob = 0 ] ; then
		# Output state
		echo Submitting first batch $currBatch
		# And submit first batch as parallel task using -t flag: each line as parallel job
		previousJob=$(fsl_sub -q $queue -t $currBatch)
	else
		# Output state 
		echo Submitting batch $currBatch which will wait for $previousJob to finish
		# And submit current batch as parallel task using -t flag, keeping it on hold until the previous batch finishes with -j flag
		previousJob=$(fsl_sub -q $queue -j $previousJob -t $currBatch)
	fi
done
