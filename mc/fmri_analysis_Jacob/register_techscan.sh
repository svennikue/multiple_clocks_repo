subjects=("01" "02")
folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30" "AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv")

scratchDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data'

# select folder where I store analysis script and .fsf file
homeDir='/Users/xpsy1114/Documents/projects/multiple_clocks'
echo Now entering the loop ....

for folder in "${folder_names[@]}"; do
    for subjectTag in "${subjects[@]}"; do
        echo Subject tag and folder for the current run: $subjectTag $folder
        # Construct directory for raw data
        rawDir=$scratchDir/tech_scan/sub-$subjectTag/$folder
        # Construct directory for derived data
        derivDir=$scratchDir/derivatives/sub-$subjectTag/$folder
        # Construct func directory for derived file
        funcDir=$derivDir/func
        # And create directory for derived functional files
        mkdir -p $funcDir

        # Get number of volumes from fslinfo and some bash tricks
        #numVols=$(echo $(echo $(fslinfo $rawDir/func/sub-${subjectTag}_bold.nii.gz) | grep -o -E ' dim4 [0-9]+' | sed 's/ dim4 //'))
        numVols=$(fslval $funcDir/sub-${subjectTag}_bold.nii.gz dim4)

        # compute the number of voxels
        dim1=$(fslval $funcDir/sub-${subjectTag}_bold.nii.gz dim1)
        dim2=$(fslval $funcDir/sub-${subjectTag}_bold.nii.gz dim2)
        dim3=$(fslval $funcDir/sub-${subjectTag}_bold.nii.gz dim3)
        dim4=$(fslval $funcDir/sub-${subjectTag}_bold.nii.gz dim4)
        numVoxels=$((dim1*dim2*dim3*dim4))

        # really important to here take the respective .fsf file of the sequence I prepared manually! Different TE, TR, echo spacing
        cat $homeDir/fmri_analysis/templates/$folder/preproc.fsf | sed "s/371195136/${numVoxels}/g" |sed "s/442/${numVols}/g" | sed "s/sub-01/sub-$subjectTag/g" | sed "s/AB_sequ_min30/$folder/g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data:${scratchDir}:g" > $funcDir/sub-${subjectTag}_${folder}_design_preproc.fsf
        feat $funcDir/sub-${subjectTag}_${folder}_design_preproc.fsf
    done
done





# # so for some reason, for subject 1, AB_sequ_plu30 and Leip_sequ_transv it didnt work.
# # therefore I am temporarily chaning the folder so it does these two again!
# #subjects=("01")
# #folder_names=("Leip_sequ_transv" "AB_sequ_plus30")


# # do everything for the 7T dataset
# subjects_7T=("02")
# folders_7T=("run1_T7_AB_rep" "run2_T7_2mm_3iPat_2MB" "run3_T7_15mm_3iPat_3MB" "run4_T7_15mm_2iPat_3MB" "run5_T7_superfast" "run6_T7_12mm_3iPat_3MB")
# scratchDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data'
# calpendoDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data/F7T_2013_40_867'


# scratchDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data'

# # select folder where I store analysis script and .fsf file
# homeDir='/Users/xpsy1114/Documents/projects/multiple_clocks'

# # ok actually, for the 7T, I decided to go with the GUI since there are so 
# # many variables that I have to manually replace from the .fsf file.

# for folder in "${folder_names[@]}"; do
#     for subjectTag in "${subjects[@]}"; do
#         # Construct directory for raw data
#         rawDir=$scratchDir/tech_scan/sub-$subjectTag/$folder
#         # Construct directory for derived data
#         derivDir=$scratchDir/derivatives/sub-$subjectTag/$folder
#         # Construct func directory for derived file
#         funcDir=$derivDir/func
#         # And create directory for derived functional files
#         mkdir -p $funcDir

#         # Get number of volumes from fslinfo and some bash tricks
#         #numVols=$(echo $(echo $(fslinfo $rawDir/func/sub-${subjectTag}_bold.nii.gz) | grep -o -E ' dim4 [0-9]+' | sed 's/ dim4 //'))
#         numVols=$(fslval $funcDir/sub-${subjectTag}_bold.nii.gz dim4)

#         #TR= #this is just depending on the 

#         # check which sequence this is and adjust the parameters
#         #echo_spacing= #is probably something of the below
#         #TR= #set fmri(npts) 1.240
#         #EPIdwell= #fmri(dwell) 0.66 BUT fmri(noise) 0.66 as well...
#         # replace structural
#         # replace fieldmap
#         # replace bold
#         # total voxels (?)
#         # replace TE set fmri(te) 22
#         # set fmri(noisear) 0.34


#         # Take preprocessing template, replace subject id and number of volumes with current values and save to new file
#         #cat $homeDir/fmri_analysis/templates/$folder/preproc.fsf | sed "s/442/${numVols}/g" | sed "s/sub-01/sub-$subjectTag/g" | sed "s/AB_sequ_min30/$folder/g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data:${scratchDir}:g" > $funcDir/sub-${subjectTag}_${folder}_design_preproc.fsf
#         # this bit replaces the following things in the fsf file:
#         # number of Volumes (numVols), subject number (subjectTag), folder name (folder)
#         # it needs to replace more stuff.

#         # replace:
#         # 1. subject 2. folder (structural, functional, fieldmap), 3. numVols, 4. TR, 5. TE, 6. numVox, 7. fmridwell
#         # dwellplaceholder TRplaceholder, TEdplaceholder, numVolplaceholder, numVoxplaceholder

#         # I think that fmri(dwell) is the echo time between two echo pulses divided by accelartion factor



#         # Finally: run feat with these parameters
#         #feat $funcDir/sub-${subjectTag}_${folder}_design_preproc.fsf

#     done
# done


# # do it just for AB_sequ_transv sub 02 again
# #numVols=$(fslval /Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-02/AB_sequ_transv/func/sub-02_bold.nii.gz dim4)
# #cat /Users/xpsy1114/Documents/projects/multiple_clocks/fmri_analysis/templates/AB_sequ_transv/preproc.fsf | sed "s/442/${numVols}/g" | sed "s/sub-01/sub-02/g" | sed "s/AB_sequ_min30/AB_sequ_transv/g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data:/Users/xpsy1114/Documents/projects/multiple_clocks/data:g" > /Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-02/AB_sequ_transv/func/sub-02_AB_sequ_transv_design_preproc.fsf
# #feat /Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-02/AB_sequ_transv/func/sub-02_AB_sequ_transv_design_preproc.fsf
