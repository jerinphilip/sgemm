#!/bin/bash

BATCH_SIZE=100
for PROVIDER in EIGEN RUY MKL BLAS; do
  echo "+ MARIAN_SGEMM_PROVIDER=$PROVIDER ./benchmark --batchSize $BATCH_SIZE"
  time MARIAN_SGEMM_PROVIDER=$PROVIDER ./benchmark --batchSize $BATCH_SIZE || echo "Failed on ${PROVIDER}. Expected?"
done
