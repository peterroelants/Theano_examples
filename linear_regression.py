import random
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T
from theano import function
from theano.sandbox.linalg import ops as linOps


###########################################################
# Data and plot functions
###########################################################
def generateRandomSampleData():
    """
    Generate random data around the line y = b + ax.
    With: b=2 and a = 0.5
    Out:
        tuple (xs, ys)
            xs: x-data with intercept for the regression.
                Each row is one sample.
            ys: y-data
                Each row is one sample.
            xs[i] corresponds to ys[i]
    """
    x_min = 0
    x_max = 10
    b = 2
    a = 0.5
    sigma = 0.5
    xs = []
    ys = []
    for i in range(20):
        x = random.uniform(x_min, x_max)
        y_mu = b + a * x
        y = random.normalvariate(y_mu, sigma)
        xs.append([1, x])
        ys.append([y])
    return (np.array(xs), np.array(ys))

def plotData(xs, ys, b1, b2):
    # plot data points
    plt.plot(xs[:,1], ys, 'b+')
    # plot orginal line
    x_line = np.linspace(0, 10)
    y_line = 2 + x_line * 0.5
    plt.plot(x_line, y_line, 'b', label='original')
    # plot line fitted by first method
    y_line = b1[0][0] + x_line * b1[1][0]
    plt.plot(x_line, y_line, 'g', label='regression 1')
    # plot line fitted by second method
    y_line = b1[0][0] + x_line * b2[1][0]
    plt.plot(x_line, y_line, 'r', label='regression 2')
    plt.show()


###########################################################
# Regression functions
###########################################################
"""
General linear model:
---------------------
y_i = b_0  +  b_1 * X_i_1  +  b_2 * X_i_2  +  ...  +  b_{p-1} * X_i_{p-1}  +  e_i
for i = 1 .. n

y = X*b + e

n = number of observations
p = number of input dimensions

X_i_j: nxp matrix (input matrix )
    i: row      ->  observations (1 .. n)
    j: column   ->  input params (0 .. p-1)
                    X_i_0 is the intercept term (= 1 if an intercept is needed)

y_i: column vector (output vector for observations)
    i: row      ->  observations (1 .. n)

b_j: column vector (regression slopes for each input dimension)
    j: row      ->  input params (0 .. p-1)
                    b_0 is the intercept if X_i_0 = 1

e_i: column vector (random errors for each observation)
    i: row      -> observations (1 .. n)
"""

def linearRegression(inputs, outputs):
    """
    Computers the least squares estimator (LSE) B_hat that minimises the sum of the
    squared errors.

    Computes B_hat as B_hat = (X.T . X)^-1 . X.T . y

    In:
        inputs: Matrix of inputs (X)
                format: [[observation_1], ..., [observation_n]]
        outputs: Column vector (Matrix) of outputs y
                 format: [[y_1], ... , [y_n]]
    Out:
        B_hat: Column vector (Matrix) of fitted slopes
               format: [[b_0], ... , [b_{p-1}]]
    """
    return linearRegression_1(inputs, outputs)


def linearRegression_1(inputs, outputs):
    """
    Computers the least squares estimator (LSE) B_hat that minimises the sum of the
    squared errors.

    In:
        inputs: Matrix of inputs (X)
                format: [[observation_1], ..., [observation_n]]
        outputs: Column vector (Matrix) of outputs y
                 format: [[y_1], ... , [y_n]]
    Out:
        B_hat: Column vector (Matrix) of fitted slopes
               format: [[b_0], ... , [b_{p-1}]]
    """
    X = T.dmatrix('X')
    y = T.dmatrix('y')
    # B_hat = (X.T . X)^-1 . X.T . y
    # http://deeplearning.net/software/theano/library/sandbox/linalg.html
    # MatrixInverse is the class.
    # matrix_inverse is the method base upon the MatrixInverse class.
    B_hat = T.dot(T.dot(linOps.matrix_inverse(T.dot(X.T, X)),X.T),y)
    lse = function([X, y], B_hat)
    b = lse(inputs, outputs)
    return b


def linearRegression_2(inputs, outputs):
    """
    Computers the least squares estimator (LSE) B_hat that minimises the sum of the
    squared errors.

    Computes B_hat as B_hat = X^+ . y
    with X^+ the pseudoinverse of matrix X.
    http://en.wikipedia.org/wiki/Moore-Penrose_pseudoinverse

    In:
        inputs: Matrix of inputs (X)
                format: [[observation_1], ..., [observation_n]]
        outputs: Column vector (Matrix) of outputs y
                 format: [[y_1], ... , [y_n]]
    Out:
        B_hat: Column vector (Matrix) of fitted slopes
               format: [[b_0], ... , [b_{p-1}]]
    """
    X = T.dmatrix('X')
    y = T.dmatrix('y')
    # http://deeplearning.net/software/theano/library/sandbox/linalg.html
    # MatrixPinv is the class.
    # pinv is the method based upon the MatrixPinv class.
    # B_hat = X^+ . y
    B_hat = T.dot(linOps.pinv(X),y)
    lse = function([X, y], B_hat)
    b = lse(inputs, outputs)
    return b


###########################################################
# main
###########################################################
def main():
    # Set random seed to make experiment repeatable.
    random.seed(1)
    xs, ys = generateRandomSampleData()
    b1 = linearRegression_1(xs, ys)
    b2 = linearRegression_2(xs, ys)
    plotData(xs, ys, b1, b2)


if __name__ == "__main__":
    main()