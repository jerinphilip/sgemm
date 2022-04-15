#!/bin/bash

BATCH_SIZE=100
for PROVIDER in eigen ruy mkl blas; do
  echo "+" ./benchmark --batchSize $BATCH_SIZE --provider $PROVIDER
  time ./benchmark --batchSize $BATCH_SIZE --provider $PROVIDER || echo "Failed on ${PROVIDER}. Expected?"
done
