
# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
#fslDir="${fslDir}"
export fslDir=~/scratch/fsl
export PATH=${fslDir}/share/fsl/bin/:$PATH
source ${fslDir}/etc/fslconf/fsl.sh

# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  analysisDir="/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo/mc/fmri_analysis"
  fslDir="/Users/xpsy1114/fsl"
fi

echo Now entering the loop ....
echo fslDir ist ${fslDir}


for subjectTag in 25 26; do
    # probably first navigate to the feat dir


    /home/fs0/xpsy1114/scratch/fsl/bin/fnirt --iout=highres2standard_head --in=highres_head --aff=highres2standard.mat --cout=highres2standard_warp --iout=highres2standard --jout=highres2highres_jac --config=/home/fs0/xpsy1114/scratch/fsl/etc/flirtsch/T1_2_MNI152_2mm.cnf --ref=standard_head --refmask=standard_mask --warpres=10,10,10
    
    ${fslDir}/bin/fnirt --iout=highres2standard_head --in=highres_head --aff=highres2standard.mat --cout=highres2standard_warp --iout=highres2standard --jout=highres2highres_jac --config=T1_2_MNI152_2mm --ref=standard_head --refmask=standard_mask --warpres=10,10,10


    ${fslDir}/bin/applywarp -i highres -r standard -o highres2standard -w highres2standard_warp


    ${fslDir}/bin/convert_xfm -inverse -omat standard2highres.mat highres2standard.mat


    ${fslDir}/bin/slicer highres2standard standard -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; ${fslDir}/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png highres2standard1.png ; ${fslDir}/bin/slicer standard highres2standard -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; ${fslDir}/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png highres2standard2.png ; ${fslDir}/bin/pngappend highres2standard1.png - highres2standard2.png highres2standard.png; /bin/rm -f sl?.png highres2standard2.png


    /bin/rm highres2standard1.png


    ${fslDir}/bin/convert_xfm -omat example_func2highres.mat -concat initial_highres2highres.mat example_func2initial_highres.mat


    ${fslDir}/bin/convert_xfm -inverse -omat highres2example_func.mat example_func2highres.mat


    ${fslDir}/bin/convertwarp --ref=highres --premat=example_func2initial_highres.mat --warp1=initial_highres2highres_warp --out=example_func2highres_warp --relout


    ${fslDir}/bin/applywarp --ref=highres --in=example_func --out=example_func2highres --warp=example_func2highres_warp


    ${fslDir}/bin/convert_xfm -inverse -omat highres2example_func.mat example_func2highres.mat


    ${fslDir}/bin/slicer example_func2highres highres -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; ${fslDir}/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png example_func2highres1.png ; ${fslDir}/bin/slicer highres example_func2highres -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; ${fslDir}/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png example_func2highres2.png ; ${fslDir}/bin/pngappend example_func2highres1.png - example_func2highres2.png example_func2highres.png; /bin/rm -f sl?.png example_func2highres2.png


    /bin/rm example_func2highres1.png


    ${fslDir}/bin/applywarp -i example_func -r example_func -o example_func -w example_func2highres_warp --postmat=highres2example_func.mat


    ${fslDir}/bin/convert_xfm -omat example_func2standard.mat -concat highres2standard.mat example_func2highres.mat


    ${fslDir}/bin/convertwarp --ref=standard --premat=example_func2highres.mat --warp1=highres2standard_warp --out=example_func2standard_warp


    ${fslDir}/bin/applywarp --ref=standard --in=example_func --out=example_func2standard --warp=example_func2standard_warp


    ${fslDir}/bin/convert_xfm -inverse -omat standard2example_func.mat example_func2standard.mat


    ${fslDir}/bin/slicer example_func2standard standard -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; ${fslDir}/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png example_func2standard1.png ; ${fslDir}/bin/slicer standard example_func2standard -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; ${fslDir}/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png example_func2standard2.png ; ${fslDir}/bin/pngappend example_func2standard1.png - example_func2standard2.png example_func2standard.png; /bin/rm -f sl?.png example_func2standard2.png


    ${fslDir}/bin/imcp ../example_func ../example_func_distorted


    ${fslDir}/bin/imcp example_func ../example_func


done