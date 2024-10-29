from builtins import object
import numpy as np

from cs6353.layers import *
from cs6353.layer_utils import *


class ConvNet(object):
    """
   A simple convolutional network with the following architecture:

    [conv - bn - relu] x M - max_pool - affine - softmax
    
    "[conv - bn - relu] x M" means the "conv-bn-relu" architecture is repeated for
    M times, where M is implicitly defined by the convolution layers' parameters.
    
    For each convolution layer, we do downsampling of factor 2 by setting the stride
    to be 2. So we can have a large receptive field size.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[32, 32], filter_sizes=[7, 3],
                 num_classes=10, weight_scale=1e-3, reg=0.0,use_batch_norm=True, 
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer. It is a
          list whose length defines the number of convolution layers
        - filter_sizes: Width/height of filters to use in the convolutional layer. It
          is a list with the same length with num_filters
        - num_classes: Number of output classes
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - use_batch_norm: A boolean variable indicating whether to use batch normalization
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the simple convolutional         #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params.                                                 #
        #                                                                          #
        # IMPORTANT:                                                               #
        # 1. For this assignment, you can assume that the padding                  #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. You need to         #
        # carefully set the `pad` parameter for the convolution.                   #
        #                                                                          #
        # 2. For each convolution layer, we use stride of 2 to do downsampling.    #
        ############################################################################

        for i in range(len(num_filters)):
            if i == 0:
                self.params['W_' + str(i + 1)] = np.random.randn(num_filters[i], input_dim[0], filter_sizes[i],
                                                                filter_sizes[i]) * weight_scale
                self.params['b_' + str(i + 1)] = np.zeros(num_filters[i])
            else:
                self.params['W_' + str(i + 1)] = np.random.randn(num_filters[i], num_filters[i-1], filter_sizes[i],
                                                    filter_sizes[i]) * weight_scale
                self.params['b_' + str(i + 1)] = np.zeros(num_filters[i])
            self.params['gamma' + str(i + 1)] = np.ones(self.params['W_' + str(i + 1)].shape[0])
            self.params['beta' + str(i + 1)] = np.zeros(self.params['W_' + str(i + 1)].shape[0])

        # self.params['W1'] = np.random.randn(num_filters[0], input_dim[0], filter_sizes[0], filter_sizes[0]) * weight_scale
        # self.params['b1'] = np.zeros(num_filters[0])
        #
        # self.params['W2'] = np.random.randn(num_filters[1], num_filters[0], filter_sizes[1], filter_sizes[1]) * weight_scale
        # self.params['b2'] = np.zeros(num_filters[1])

        self.params['W_' + str(len(num_filters) + 1)] = np.random.randn(num_filters[len(num_filters) - 1] * int((input_dim[1] / 2)) * int((input_dim[2] / 2)), num_classes) * weight_scale
        self.params['b_' + str(len(num_filters) + 1)] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        scores = None
        mode = 'test' if y is None else 'train'
        ############################################################################
        # TODO: Implement the forward pass for the simple convolutional net,       #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out = None
        caches = {}
        layers = int(len(self.params) / 4)

        for i in range(layers):
            conv_param = {}
            conv_param['stride'] = 1
            conv_param['pad'] = int(
                (conv_param['stride'] * (X.shape[2] - 1) - X.shape[2] + self.params['W_' + str(i + 1)].shape[2]) / 2)
            bn_param = {'mode': 'train'}
            gamma = self.params['gamma' + str(i + 1)]
            beta = self.params['beta' + str(i + 1)]
            if i == 0:
                out, cache = conv_bn_relu_forward(X, self.params['W_' + str(i + 1)], self.params['b_' + str(i + 1)], gamma, beta,
                                                  conv_param, bn_param)
            else:
                out, cache = conv_bn_relu_forward(out, self.params['W_' + str(i + 1)], self.params['b_' + str(i + 1)], gamma,
                                                    beta,
                                                    conv_param, bn_param)
            caches['cache' + str(i + 1)] = cache

        # first_conv_param = {}
        # first_conv_param['stride'] = 1
        # first_conv_param['pad'] = int((first_conv_param['stride']*(X.shape[2] - 1) - X.shape[2] + self.params['W1'].shape[2]) / 2)
        # first_bn_param = {'mode': 'train'}
        # first_gamma = self.params['gamma1']
        # first_beta = self.params['beta1']
        # a_1, cache_1 = conv_bn_relu_forward(X, self.params['W1'], self.params['b1'], first_gamma, first_beta, first_conv_param, first_bn_param)
        #
        # second_conv_param = {}
        # second_conv_param['stride'] = 1
        # second_conv_param['pad'] = int((second_conv_param['stride']*(a_1.shape[2] - 1) - a_1.shape[2] + self.params['W2'].shape[2]) / 2)
        # second_gamma = self.params['gamma2']
        # second_beta = self.params['beta2']
        # second_bn_param = {'mode': 'train'}
        # a_2, cache_2 = conv_bn_relu_forward(a_1, self.params['W2'], self.params['b2'], second_gamma, second_beta,
        #                                     second_conv_param, second_bn_param)

        pool_param = {}
        pool_param['pool_height'] = 2
        pool_param['pool_width'] = 2
        pool_param['stride'] = 2

        a_3, pool_cache = max_pool_forward_fast(out, pool_param)

        # caches['cache' + str(layers + 1)] = pool_cache

        out, out_cache = affine_forward(a_3, self.params['W_' + str(layers + 1)], self.params['b_' + str(layers + 1)])

        # caches['cache' + str(layers + 2)] = out_cache

        # return first_conv_param['pad'], X.shape, a_1.shape, a_2.shape, a_3.shape, out.shape

        scores = out

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the simple convolutional net,      #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        dout = None
        loss, dscores = softmax_loss(scores, y)

        for i in range(layers + 2, 0, -1):
            if i == layers + 2:
                dout, grads['W_' + str(i - 1)], grads['b_' + str(i - 1)] = affine_backward(dscores, out_cache)
                grads['W_' + str(i - 1)] += self.reg * self.params['W_' + str(i - 1)]
                loss += 0.5 * self.reg * np.sum(self.params['W_' + str(i - 1)] ** 2)
            elif i == layers + 1:
                dout = max_pool_backward_fast(dout, pool_cache)
            else:
                dout, grads['W_' + str(i)], grads['b_' + str(i)], grads['gamma' + str(i)], grads['beta' + str(i)] = conv_bn_relu_backward(dout, caches['cache' + str(i)])
                grads['W_' + str(i)] += self.reg * self.params['W_' + str(i)]
                loss += 0.5 * self.reg * np.sum(self.params['W_' + str(i)] ** 2)
                
        # dout, grads['W3'], grads['b3'] = affine_backward(dscores, out_cache)
        # da_3 = max_pool_backward_fast(dout, cache_3)
        # da_2, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = conv_bn_relu_backward(da_3, cache_2)
        # da_1, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_bn_relu_backward(da_2, cache_1)

        # grads['W1'] += self.reg * self.params['W1']
        # grads['W2'] += self.reg * self.params['W2']
        # grads['W3'] += self.reg * self.params['W3']

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
