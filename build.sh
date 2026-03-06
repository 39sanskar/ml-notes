#!/bin/sh

set -xe

clang -Wall -Wextra -o linear_regression_gradient_descent linear_regression_gradient_descent.c -lm

clang -Wall -Wextra -o single_neuron_trainer single_neuron_trainer.c -lm

clang -Wall -Wextra -o xor_neural_network xor_neural_network.c -lm
