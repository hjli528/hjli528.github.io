---
published: true
---
In this post, we will talk about how logistic regression works and derive the gradient of loss with respect to the model's parameter. We will also build a logistic regression model from scratch in python.

## Logistic Regression
A logistic regression computes a weighted sum $z$ of the input features plus a bias term, and then pass the
results to a logistic function, $h_\theta (z)$.

$$h_\theta (z) = \sigma(x^T\theta+b)$$

where $x$ is the training example written in column form. $\theta$ is the model's parameter vector and $b$ is the bias term.

$$ z = \begin{bmatrix}  
        x_1 \\
        x_2 \\
        x_3 \\
        ...   \\
        x_n
    \end{bmatrix}^T
    \begin{bmatrix}  
        \theta^1 \\
        \theta^2 \\
        \theta^3 \\
        ...      \\
        \theta^n
        \end{bmatrix} + b
    $$

Often the bias term is set as $\theta_0$ and the above matrix multiplication can be rewritten as

$$ z = \begin{bmatrix}  
        1     \\
        x_1 \\
        x_2 \\
        x_3 \\
        ...   \\
        x_n
    \end{bmatrix}^T
    \begin{bmatrix}
        \theta^0 \\
        \theta^1 \\
        \theta^2 \\
        \theta^3 \\
        ...      \\
        \theta^n
        \end{bmatrix}
    $$

or $$h_\theta (z) = \sigma(\theta \cdot x)$$

We call the logistic function a hypothesis function and it is defined as

$$\sigma(z) = \frac{1}{1 + \exp(-z)}$$

![Logistic function]({{site.baseurl}}/images/logistic_function_09122021.png)

Once the logistic function outputs the probability $\hat p = h_\theta(z)$, we can make a prediction easily

$$\begin{align}
\hat y =\left\{
                \begin{array}{ll}
                     0, \text{if}\quad \hat p < 0.5\\
                     1, \text{if}\quad \hat p \ge 0.5
                \end{array}
              \right.
\end{align}$$

Notice that we have $h_\theta(z) < 0.5$ if $z < 0$, and $h_\theta(z) \ge 0.5$ if $z \ge 0$, so a logistic regression predicts 1 if $z=\theta \cdot x $ is positive and 0 if it is negative [1].

## Loss Function
For a logistic regression classification task, the goal of training is to find the parameter vector $\theta$ so that the model predicts probabilities close to 1 for positive samples ($y=1$) and probabilities close to 0 for negative samples ($y=0$). This is equivalent to minimize the following loss function

$$\begin{align}
\mathbb{l}(\theta) =\left\{
                \begin{array}{ll}
                     -\log(\hat{p}), \text{if}\quad y = 1\\
                     -\log(1-\hat{p}), \text{if}\quad y = 0
                \end{array}
              \right.
\end{align}$$

![Logistic function loss]({{site.baseurl}}/images/logistic_loss_09122021.png)

As shown in figure above, the loss $\mathbb{l}$ grows very large when the model's probability output is close to 0 for positive samples (y=1). On the other hand, the loss is close to 0 when the model's probability is close to 1.0. Conveniently, equation above can be written as

$$\mathbb{l(\theta)}=-y\log(\hat{p})-(1-y)\log(1-\hat{p})$$

$\mathbb{l(\theta)}$ is called <em>binary cross entropy</em>. For a training set with $m$ samples, the total is just the mean of $\mathbb{l(\theta)}$, i.e.,

$$L(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[y_i\log(\hat{p_i})+(1-y_i)\log(1-\hat{p_i})]$$

## Training with Gradient Descent
The next step is to use gradient descent to minimize the loss $L(\theta)$ over the whole training set. Using the chain rule, we can find the gradient of loss with respect to $\theta$

$$
\begin{aligned}
\frac{\partial L}{\partial \theta}
 & = \frac{\partial L}{\partial h_\theta}\frac{\partial h_\theta}{\partial \theta} \\
 & = -\frac{1}{m}\sum_{i=1}^{m}[\frac{y_i}{\ln(2)\hat{p_i}}+\frac{1-y_i}{-(1-\hat{p_i})\ln(2)}]\frac{\partial h_\theta}{\partial \theta} \\
 & = -\frac{1}{m\ln(2)}\sum_{i=1}^{m}[\frac{y_i}{\hat{p_i}}-\frac{1-y_i}{1-\hat{p_i}}]\frac{-1}{(1-\exp(-z))^2}-\exp(-z)\frac{\partial z}{\partial \theta} \\
 & = -\frac{1}{m\ln(2)}\sum_{i=1}^{m}[\frac{y_i}{\hat{p_i}}-\frac{1-y_i}{1-\hat{p_i}}]\hat{p_i}\frac{\exp(z)}{1+\exp(-z)}{\partial \theta} \\
 & = -\frac{1}{m\ln(2)}\sum_{i=1}^{m}[\frac{y_i}{\hat{p_i}}-\frac{1-y_i}{1-\hat{p_i}}]\hat{p_i}(1-\hat{p_i})x \\
 & = -\frac{1}{m\ln(2)}\sum_{i=1}^{m}({y_i}(1-\hat{p_i})-\hat{p_i}(1-{y_i}))x \\
 & = - \frac{1}{m\ln(2)}\sum_{i=1}^{m}(y_i - \hat{p_i})x \\
 & = \frac{1}{m\ln(2)}\sum_{i=1}^{m}(\hat{p_i} - y_i)x
\end{aligned}
$$

Note that since we set $b$ as part of $\theta$ (i.e., $b=\theta_0$), the gradient of loss with respect to $b$ is

$$
\begin{aligned}
\frac{\partial L}{\partial b} &= \frac{1}{m\ln(2)}\sum_{i=1}^{m}(\hat{p_i} - y_i)x_0 \\
& = \frac{1}{m\ln(2)}\sum_{i=1}^{m}(\hat{p_i} - y_i)
\end{aligned}
$$

Now we can use graient descent to update the $\theta$

$$
\begin{aligned}
\theta &:=\theta - \text{lr}\frac{\partial L}{\partial \theta} \\
 &= \theta - \frac{\text{lr}}{\ln(2)}\frac{1}{m}\sum_{i=1}^{m}(\hat{p_i} - y_i) \\
 & = \theta - \text{lr}\frac{1}{m}\sum_{i=1}^{m}(\hat{p_i} - y_i)
\end{aligned}
$$

where $\text{lr}$ is the learning rate.
## Matrix Form
In practical applications, training samples are passed to the algorithm all together or by <em>batch</em>. In Machine Learning, vectors are often represented as column vectors. Suppose our training set contains $m$ training examples with $n$ features, then it can be written as:
$$ X = \begin{bmatrix}
    1     & 1     & 1     & ... & 1     \\
    x_1^1 & x_2^1 & x_3^1 & ... & x_m^1 \\
    x_1^2 & x_2^2 & x_3^2 & ... & x_m^2 \\
    x_1^3 & x_2^3 & x_3^3 & ... & x_m^3 \\
    ...   & ...   & ...   & ... & ...   \\
    x_1^n & x_2^n & x_3^n & ... & x_m^n
    \end{bmatrix}, \text{with label Y =}
    \begin{bmatrix}
        y_1 & y_2 & y_3 & ... & y_m
     \end{bmatrix}$$

where $x_i^j$ represents the value of feature $j$ of sample $i$. Then the probability of all $m$ samples can be computed directly via matrix multiplication

$$ h_\theta(Z) = \sigma(\begin{bmatrix}
    1     & 1     & 1     & ... & 1     \\
    x_1^1 & x_2^1 & x_3^1 & ... & x_m^1 \\
    x_1^2 & x_2^2 & x_3^2 & ... & x_m^2 \\
    x_1^3 & x_2^3 & x_3^3 & ... & x_m^3 \\
    ...   & ...   & ...   & ... & ...   \\
    x_1^n & x_2^n & x_3^n & ... & x_m^n
    \end{bmatrix}^T
    \begin{bmatrix}
        \theta^0 \\
        \theta^1 \\
        \theta^2 \\
        \theta^3 \\
        ...      \\
        \theta^n
        \end{bmatrix})$$

Consequently, the gradient descent over the whole training set or <em>batch</em> is

$$
\begin{aligned}
\theta &:=\theta - \text{lr}\frac{\partial L}{\partial \theta} \\
 & = \theta - \text{lr}\frac{1}{m}(h_\theta(Z) - Y)
\end{aligned}
$$

## Implementation from Scratch
```
# Implement logistic regression from scratch
# 09/12/2021
# Implementation based on the work of
# https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2
# and https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc

import numpy as np
import matplotlib.pyplot as plt

# Randomly generate some data for binary classifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features = 2, n_redundant=0,
                           n_informative=2, random_state=1,
                           n_clusters_per_class= 1)
print("number of training samples {} and features {}".format(X.shape[0], X.shape[1]))

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def loss(y, y_hat):
    loss = -np.mean(y*np.log(y_hat) + (1.0-y)*np.log(1.0-y_hat))
    return loss

def gradients(X, y, y_hat):
    # X : Input, m_sample X n_feature
    # y : Ground truth label m_sample X 1
    # y_hat : predictions m_sample X 1
    # w : model's parameters 1 X n_feature
    # b : bias
    m = X.shape[0]
    # Gradients of loss with respect to the weights w
    dw = (1/m)*np.dot(X.T, (y_hat - y))

    # Gradients of loss with respect to the bias
    db = (1/m)*np.sum(y_hat - y)

    grads = {"dw":dw, "db":db}

    return grads

def plot_decision_boundary(X, w, b):
    # X : Input, m_sample X n_feature
    # w : model's parameters 1 X n_feature
    # b : bias

    # The decision boundary is a line y_db = kx + x0
    # For logistic regression with 2 features, we have z = b + w1x1 + w2x2 = 0
    # or x2 = -b/w2 - w1/w2 x

    x_lim = [min(X[:, 0]), max(X[:, 1])]
    m = -w[0]/w[1]
    c = -b/w[1]
    y_lim = m*x_lim + c

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    # Samples with label 0
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "g^")
    # Samples with label 1
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
    plt.xlim([-2, 2])
    plt.ylim([0, 2.2])
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title("decision boundary")

    plt.plot(x_lim, y_lim, 'y-')
    plt.show()
    plt.savefig("logistic_regression_decision_boundary.png")

def normalize(X):
    # X : Input, m_sample X n_feature
    # m : number of training samples
    # n : number of features
    m, n = X.shape
    for i in range(n):
        X = (X - X.mean(axis=0))/X.std(axis=0)

    return X

def train(X, y, bs, epochs, lr):
    # X : Input, m_sample X n_feature
    # y : Ground truth label m_sample X 1
    # bs : Batch size
    # epochs: number of epochs
    # lr : learning rate

    m, n = X.shape

    w = np.zeros((n, 1))
    print(w.shape)
    b = 0

    y = y.reshape(m, 1)

    x = normalize(X)

    losses = []

    # training loop
    for epochs in range(epochs):
        for i in range((m-1)//bs + 1):

            start_i = i *bs
            end_i = start_i + bs
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]

            y_hat = sigmoid(np.dot(xb, w) + b)

            grads = gradients(xb, yb, y_hat)
            dw = grads["dw"]
            db = grads["db"]

            w -=lr*dw
            b -=lr*db

        l = loss(y, sigmoid(np.dot(X, w) + b))
        losses.append(l)
    # Plot the loss against time
    plt.plot(losses)
    plt.xlabel("epochs")
    plt.ylabel("loss (-)")
    plt.show()
    plt.savefig("logistic_regression_loss_vs_epochs.png")
    return w, b, losses

def predict(X):
    X = normalize(X)

    z = sigmoid(np.dot(X, w) + b)

    pred_class = []

    pred_class = [1 if i >= 0.5 else 0 for i in z]

    return np.array(pred_class)

def accuracy(y, y_hat):
    accuracy = np.sum(y == y_hat) / len(y)
    return accuracy

# Training
w, b, l = train(X, y, bs=100, epochs=1000, lr=0.01)

plot_decision_boundary(X, w, b)

print("Prediction accuracy {}".format(accuracy(y, predict(X))))

print("done!")
```
For a randomly generated training set from sklearn, the training losses gradually decreases to 0
![Logistic function loss vs time]({{site.baseurl}}/images/logistic_regression_loss_vs_epochs_09122021.png)

The corresponding decision boundary is
![Logistic function decision boundary]({{site.baseurl}}/images/logistic_regression_decision_boundary_09122021.png)

The prediction accuray for this simple classifier is 100%.

## References
1.
Geron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras and TensorFlow: concepts, tools, and techniques to build intelligent systems (2nd ed.). O’Reilly.
