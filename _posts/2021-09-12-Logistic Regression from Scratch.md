---
published: true
---
## Logistic regression
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

![Logistic function]({{site.baseurl}}/images/logistic_loss_09122021.png)

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

## References
1.
Geron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras and TensorFlow: concepts, tools, and techniques to build intelligent systems (2nd ed.). O’Reilly.
