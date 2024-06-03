#!/bin/bash

NEWFSLDIR=/opt/fmrib/fsltmp/featTestRick
module rm fsl/current
module rm sge
export FSLDIR=${NEWFSLDIR}
export PATH=${FSLDIR}/share/fsl/bin:${PATH}
. ${FSLDIR}/etc/fslconf/fsl.sh
export FSLSUB_CONF=/opt/fmrib/fsl_sub_config/fsl_sub_2.5-1.0.yml
