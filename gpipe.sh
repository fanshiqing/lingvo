#!/bin/bash
# Model file:
# lingvo/tasks/lm/params/one_billion_wds.py

set -x
export CUDA_VISIBLE_DEVICES=4
num_gpu=1
num_micro=1
trans_layers=3
logdir=/home/shiqing.fsq/shiqing.fsq/lingvo/log/lm/transformer_layer_3_gpu_${num_gpu} 
rm -rf ${logdir}/train
./bazel-bin/lingvo/trainer \
  --run_locally=gpu --mode=sync \
  --enable_asserts=false \
  --model=lm.one_billion_wds.OneBWdsGPipeTransformerWPM \
  --logdir=${logdir} \
  --logtostderr \
  --worker_split_size=${num_gpu} \
  --worker_gpus=${num_gpu} \
  --saver_keep_checkpoint_every_n_hours=10000 \
  > transformer_layers_${trans_layers}_gpu${num_gpu}_num_micro_${num_micro}_log.txt 2>&1 &
 # --controller_gpus=${num_gpu} \

  #--controller_gpus=2 \
