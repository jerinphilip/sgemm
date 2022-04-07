#!/bin/bash

BATCH_SIZE=1000
for PROVIDER in eigen ruy mkl blas; do
  echo "+" ./benchmark --batchSize $BATCH_SIZE --provider $PROVIDER
  time ./benchmark --batchSize $BATCH_SIZE --provider $PROVIDER
done
