from builtins import range
from builtins import object
import numpy as np

from cs6353.layers import *
from cs6353.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################

        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        a_1, cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        a_2, fc_cache_2 = affine_forward(a_1, self.params['W2'], self.params['b2'])

        scores = a_2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss, dscores = softmax_loss(scores, y)
        dx_2, grads['W2'], grads['b2'] = affine_backward(dscores, fc_cache_2)
        dx_1, grads['W1'], grads['b1'] = affine_relu_backward(dx_2, cache)

        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']

        loss += 0.5 * self.reg * np.sum(self.params['W1'] ** 2) + 0.5 * self.reg * np.sum(self.params['W2'] ** 2)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deterministic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################

        all_layer_dim = [input_dim] + hidden_dims + [num_classes]
        for i in range(1, len(all_layer_dim)):
            self.params['W' + str(i)] = np.random.normal(scale=weight_scale, size=(all_layer_dim[i-1], all_layer_dim[i]))
            self.params['b' + str(i)] = np.zeros(all_layer_dim[i])

            if (self.normalization == "batchnorm" or self.normalization == "layernorm") and i != len(all_layer_dim) - 1:
                self.params['gamma' + str(i)] = np.ones(hidden_dims[i - 1])
                self.params['beta' + str(i)] = np.zeros(hidden_dims[i - 1])


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        caches = {}
        caches_batchnorm = {}
        a = X

        for i in range(1, self.num_layers+1):
            # cache = 'cache_' + str(i)
            if i != self.num_layers:

                if self.normalization=='batchnorm':

                    a, f_cache = affine_forward(a, self.params['W' + str(i)], self.params['b' + str(i)])
                    a, b_cache = batchnorm_forward(a, self.params['gamma' + str(i)], self.params['beta' + str(i)], self.bn_params[i - 1])
                    a, r_cache = relu_forward(a)
                    cache = (f_cache, b_cache, r_cache)

                elif self.normalization=='layernorm':

                    a, f_cache = affine_forward(a, self.params['W' + str(i)], self.params['b' + str(i)])
                    a, b_cache = layernorm_forward(a, self.params['gamma' + str(i)], self.params['beta' + str(i)],
                                                   self.bn_params[i - 1])
                    a, r_cache = relu_forward(a)
                    cache = (f_cache, b_cache, r_cache)

                else:
                    a, cache = affine_relu_forward(a, self.params['W' + str(i)], self.params['b' + str(i)])

                if self.use_dropout:
                    a, caches['drop_' + str(i)] = dropout_forward(a, self.dropout_param)

            else:
                a, cache = affine_forward(a, self.params['W' + str(i)], self.params['b' + str(i)])

            caches['cache_' + str(i)] = cache

        scores = a

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss, dscore = softmax_loss(scores, y)
        da = None
        for i in range(self.num_layers, 0, -1):

            if i != self.num_layers:

                if self.use_dropout:
                    d_cache = caches['drop_' + str(i)]
                    da = dropout_backward(da, d_cache)

                if self.normalization=='batchnorm':

                    f_cache, b_cache, r_cache = caches['cache_' + str(i)]
                    da = relu_backward(da, r_cache)
                    da, grads['gamma' + str(i)], grads['beta' + str(i)] = batchnorm_backward(da, b_cache)
                    da, grads['W' + str(i)], grads['b' + str(i)] = affine_backward(da, f_cache)

                elif self.normalization=='layernorm':

                    f_cache, b_cache, r_cache = caches['cache_' + str(i)]
                    da = relu_backward(da, r_cache)
                    da, grads['gamma' + str(i)], grads['beta' + str(i)] = layernorm_backward(da, b_cache)
                    da, grads['W' + str(i)], grads['b' + str(i)] = affine_backward(da, f_cache)

                else:
                    da, grads['W' + str(i)], grads['b' + str(i)] = affine_relu_backward(da, caches['cache_' + str(i)])

            else:
                da, grads['W' + str(i)], grads['b' + str(i)] = affine_backward(dscore, caches['cache_' + str(i)])

            grads['W' + str(i)] += self.reg * self.params['W' + str(i)]
            loss += 0.5 * self.reg * np.sum(self.params['W' + str(i)] ** 2)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
