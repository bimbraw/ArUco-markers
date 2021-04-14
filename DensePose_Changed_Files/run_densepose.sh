#!/bin/bash

source /home/keshav/anaconda2/bin/activate

python2 /home/keshav/DensePose/tools/infer_simple_original.py \
--cfg /home/keshav/DensePose/configs/DensePose_ResNet101_FPN_32x8d_s1x-e2e.yaml \
--output-dir /home/keshav \
--image-ext jpeg \
--wts /home/keshav/DensePose_Code/lung_ultrasound/DensePose_ResNet101_FPN_s1x-e2e.pkl \
/home/keshav/DensePose_Code/image-007.jpeg

