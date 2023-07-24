#!/bin/bash

CHECKPOINT_FILE="/projects/tang/fsg/preprocess/outputs/MPIIGaze.h5"

CMD=""
CMD="main.py \
    --checkpoint-file ${CHECKPOINT_FILE} \
    --mode train \
    --model vgg-like \
    --evaluation ROC\"
    
    eval "python $CMD "
    
