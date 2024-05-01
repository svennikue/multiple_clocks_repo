# this script copies stuff from folder from laptop to the correct folder on the server.

scratchDir="/home/fs0/xpsy1114/scratch"

laptopDir=$scratchDir/from_laptop

dataPilotDir=$scratchDir/data/pilot
dataDerivDir=$scratchDir/data/derivatives


subjects=("02" "03" "04" "05" "06")

for subjectTag in "${subjects[@]}"; do
    echo "now looking at folder: $dataPilotDir/sub-${subjectTag}"
    cp ${laptopDir}/pilot/sub-${subjectTag}/beh/* ${dataPilotDir}/sub-${subjectTag}/beh/
    cp -r ${laptopDir}/derivatives/sub-${subjectTag}/func/* ${dataDerivDir}/sub-${subjectTag}/func/
done