import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1]
  all_scorse = X.dot(W)
  for index in range(all_scorse.shape[0]):
    correct_class_score = 0
    all_class_score = 0
    for i in range(num_classes):
      if i == y[index]:
        correct_class_score = np.exp(all_scorse[index, i])
        all_class_score += np.exp(all_scorse[index, i])
      else:
        all_class_score += np.exp(all_scorse[index, i])
    loss += -np.log(correct_class_score / all_class_score)
    for i in range(num_classes):
      if i == y[index]:
        dW[:, i] += ((correct_class_score / all_class_score) - 1) * X[index]
      else:
        dW[:, i] += (np.exp(all_scorse[index, i]) / all_class_score) * X[index]
  loss /= X.shape[0]
  loss += reg * np.sum(W * W)
  dW /= X.shape[0]
  dW += 2 * reg * W



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  all_scorse = X.dot(W)
  num_train = X.shape[0]
  all_scores_exp = np.exp(all_scorse)
  all_correct_exp = all_scores_exp[np.arange(all_scores_exp.shape[0]), y]
  loss = np.sum(-np.log(all_correct_exp / np.sum(all_scores_exp, axis=1)))
  loss /= X.shape[0]
  loss += reg * np.sum(W * W)
  score_to_w = all_scores_exp / np.sum(all_scores_exp, axis=1).reshape(-1, 1)
  score_to_w[np.arange(score_to_w.shape[0]), y] = score_to_w[np.arange(score_to_w.shape[0]), y] - 1
  dW = X.T.dot(score_to_w)
  dW /= X.shape[0]
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

