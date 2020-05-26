# bazel build -c opt //lingvo:trainer
# bazel build -c opt --config=cuda //lingvo:trainer
set -x
rm -rf /tmp/mnist/log/train
CUDA_VISIBLE_DEVICES=4 ./bazel-bin/lingvo/trainer \
  --run_locally=gpu \
  --mode=sync \
  --model=image.mnist.LeNet5 \
  --logdir=/tmp/mnist/log \
  --saver_max_to_keep=2 \
  --logtostderr > output.log 2>&1 &
