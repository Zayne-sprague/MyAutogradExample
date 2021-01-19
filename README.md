# MyAutogradExample

Simple example of Numerical Auto Differentiation inspired by PyTorch and Autograd

This is a full implementation of the Backpropagation algorithm used in a simple Feed Forward Neural Network that can achieve 65-75% prediction accuracy on the MINST
validation data set.

Backpropagation at the moment is very easily saturated, so early cut
off and checkpointing is almost necessary for achieving a good model.

This should not be used for anything beyond academic / personal interest.

FNN.py when ran, will train a Feed Forward Network on the MINST dataset using
the custom tensor class + backprop aglorithm to train the network.

