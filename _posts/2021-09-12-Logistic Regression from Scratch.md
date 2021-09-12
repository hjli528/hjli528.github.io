---
published: true
---
## Logistic regression
A logistic regression computes a weighted sum $z$ of the input features plus a bias term, and then pass the
results to a logistic function, $h_\theta (z)$.

$$h_\theta (z) = \sigma(x^T\theta+b)$$

where $x$ is the training example written in column form. $\theta$ is the model's parameter vector and $b$ is the bias term.

$$ z = \begin{bmatrix}  
    x_i^1 \\
    x_i^2 \\
    x_i^3 \\
    ...   \\
    x_i^m
    \end{bmatrix}^T
    \begin{bmatrix}  
        \theta^1 \\
        \theta^2 \\
        \theta^3 \\
        ...   \\
        \theta^m
        \end{bmatrix} + b
    $$

Often the bias term is set as $\theta_0$ and the above matrix multiplication can be rewritten as

$$ z = \begin{bmatrix}  
    1     \\
    x_i^1 \\
    x_i^2 \\
    x_i^3 \\
    ...   \\
    x_i^m
    \end{bmatrix}^T
    \begin{bmatrix}
        \theta^0 \\
        \theta^1 \\
        \theta^2 \\
        \theta^3 \\
        ...   \\
        \theta^m
        \end{bmatrix}
    $$

or $h_\theta (z) = \theta \cdot x$

We call the logistic function our hypothesis function and it is defined as

$h_\theta(z) = \frac{1}{1 + \exp(-z)}$

Once the logistic function outputs the probability $\hat p = h_\theta(z)$, we can make a prediction easily

f(x)=\begin{cases}1 & x\in\mathbb{Q}\\ 0 & x\notin\mathbb{Q} \\ \end{cases}

##
In Machine Learning, vectors are often represented as column vectors. Suppose our training set contains $n$ training examples with $m$ features, then it can be written as:
$$ X = \begin{bmatrix}
    x_1^1 & x_2^1 & x_3^1 & ... & x_n^1 \\
    x_1^2 & x_2^2 & x_3^2 & ... & x_n^2 \\
    x_1^3 & x_2^3 & x_3^3 & ... & x_n^3 \\
    ...   & ...   & ...   & ... & ...   \\
    x_1^m & x_2^m & x_3^m & ... & x_n^m
    \end{bmatrix}$$

where $x_i^j$ represents the value of feature $j$ of sample $i$.


## References
1.
Geron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras and TensorFlow: concepts, tools, and techniques to build intelligent systems (2nd ed.). O’Reilly.
