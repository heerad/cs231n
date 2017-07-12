import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
    
    f_i = X[i].dot(W)
    f_i -= np.max(f_i)

    f_i_exp = np.exp(f_i)
    p_i = f_i_exp / np.sum(f_i_exp)

    loss -= np.log(p_i[y[i]])
    
    for k in xrange(num_classes):
      dW[:,k] += (p_i[k] - (y[i] == k)) * X[i]
    
  loss /= num_train
  loss += reg * np.sum(W * W)
  
  dW /= num_train
  dW += 2*reg*W
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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  f = X.dot(W)
  f -= np.max(f ,axis=1, keepdims=True)
  f_exp = np.exp(f)
  p = f_exp / np.sum(f_exp, axis=1, keepdims=True)
  
  loss = np.mean(-np.log(p[np.arange(num_train),y])) + reg*np.sum(W*W)
  
  indic = np.zeros_like(p)
  indic[np.arange(num_train),y] = 1
  
  dW = (X.T.dot(p - indic))/num_train + 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

