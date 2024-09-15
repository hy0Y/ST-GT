#!/bin/bash

DATA_ID=17YKJp1z2WuHjASW2rGqn6jJE7p0dYs1k
FNAME=preproc.tar.gz

gdown --id ${DATA_ID}
tar -zxvf ${FNAME}
rm ${FNAME}