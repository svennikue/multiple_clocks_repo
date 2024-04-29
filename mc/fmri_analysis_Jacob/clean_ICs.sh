#!/bin/sh
# After classifying ICs as noise, signal, or unkown, this script regresses out the noise components and then does spatial smoothing

subjects="01"
# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
toolboxDir="/home/fs0/xpsy1114/scratch/analysis"
# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  toolboxDir="/Users/xpsy1114/Documents/toolboxes"
fi


for subjectTag in "${subjects[@]}"; do
    # Show what ended up being the scratch directory and which subject we are working on 
    echo Scratch directory is $scratchDir
    echo Subject tag for this subject: $subjectTag


    # Construct directory for derived data
    derivDir=$scratchDir/derivatives/sub-$subjectTag
    # Set directory containing previously run preprocessing
    preprocDir=$derivDir/func/preproc_clean_02.feat

    echo "starting a log file" > $preprocDir/logging_fix.txt
    # Use labeled ICA components and regress out their contribution; also regress out motion regressors
    # use -a flag for ica cleanup and -m for motion correction.
    # folder needs to be .ica folder.
    # $toolboxDir/fix/fix -a $preprocDir/filtered_func_data.ica/labels.txt -m
    # instead of doing fix, just do fsl_regfilt, since I am doing manual IC selection anyways.
    prep_noise_ICs=$(tail -n 1 $preprocDir/filtered_func_data.ica/labels.txt)  
    noise_ICs=$(echo "$prep_noise_ICs" | tr -d '[]' | sed 's/\t/,/g' | sed 's/\(.*\)/"\1"/') 

    fsl_regfilt -i $preprocDir/filtered_func_data.nii.gz -d $preprocDir/filtered_func_data.ica/melodic_mix -f $noise_ICs -o $preprocDir/filtered_func_data_clean.nii.gz

    # Display progress to see if any error messages are caused by fix or by later commands
    echo Finished denoising using labels in $preprocDir/labels.txt >> $preprocDir/logging_denoise.txt

    # Get median value of data
    m=$(fslstats $preprocDir/filtered_func_data_clean.nii.gz -k $preprocDir/mask.nii.gz -p 50)

    # Multiply median by 0.75 to get brigthness threshold for susan
    bth=$(echo 0.75*$m | bc)

    # Get sigma for gaussian, given the FWHM of 5.0mm: multiply FWHM by 1/(2*sqrt(2*ln(2))) = 0.424660900144010
    s=$(echo 0.424660900144010*5.0 | bc)

    # Check if these were successful
    echo Running SUSAN smoothing with threshold $bth, calculated from median $m, and sigma $s

    # Do smoothing for later volumetric analysis
    susan $preprocDir/filtered_func_data_clean.nii.gz $bth $s 3 1 1 $preprocDir/mean_func.nii.gz $bth $preprocDir/filtered_func_data_clean_smooth.nii.gz

    # Remove some resulting files - the usan file is particularly big
    rm $preprocDir/filtered_func_data_clean_smooth_usan_size.nii.gz $preprocDir/filtered_func_data_clean_vn.nii.gz
done