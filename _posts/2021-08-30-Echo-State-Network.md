---
published: true
---
## What is Echo State Network?

Echo State Network (ESN) is a class of reservoir computing and is considered as partially-trained neural networks. Three main components of ESN are input, reservoir, and output, as shown in figure below [1]. Unlike conventional neural networks, the input weight matrix $W_{in} \in \mathbb{R}^{n_r \times n_{in}}$ and reservoir layer weight matrix $W_{x} \in \mathbb{R}^{n_r \times n_r}$ are randomly generated and never changed during training or testing phases of the network. These are typically sparse matrices constructed so that the reservoir verifies the Echo State Property. The output layer linearly combines the desired output signal from the high-dimensional hidden layer, and its weights $W_{out} \in \mathbb{R}^{n_{out} \times n_r}$ are trained during training process.

![echo_state_network]({{ site.baseurl }}/images/Echo_State_Network_2021_08_30.png)

## Training Process of Echo State Network
Suppose we have $m$ training vector pairs, where $u[n]$ is the n-th input vector with a size of $(n_{in} \times 1)$ and $y[n]$ is the corresponding output vector with a size of $(n_{out} \times 1)$.

The training steps of ESN are:

1. At initialization, randomly generate the weights $W_{in}$ and $W_x$ for the input and reservoir layers, respectively.

2. Feed the next input vector $u[n+1]$ to the input layer

3. Calculate the response of the reservoir layer using

	$x[n+1] = f^{res} (W_{in} u[n+1] + W_x x[n])$

	where $f^{res}$ is the reservoir layer's activation function. Let's check the dimension of both sides of the equation

	$(n_r \times 1) = f^{res}((n_r \times n_{in})\times(n_{in} \times 1) + (n_r \times n_r)(n_r \times 1))$ $\checkmark$

4. Save the response $x[n+1]$ in a matrix $X$ (concatenating by column).

5. Repeat steps 2 - 4 for all $m$ training pairs

6. Calculate output weights $W_{out}$ based on the equation below

	$W_{out} = (YX')(XX')^{-1}$

	Let's check the dimensions of both sides of the above equation

	$(n_{out} \times n_r) = \left((n_{out}\times m)(m \times n_r)\right)\left((n_r \times m)(m \times n_r)\right)^{-1} = (n_{out} \times n_r)$ $\checkmark$

7. Once the output weight matrix $W_{out}$ is calculated, the network is ready and the state of the reservoir layer is used to calculate the output of the network as
	$y[n+1] = f^{out}(W_{out}x[n+1])$

Note that Step 6 means that no gradient-based optimization is used to calculate $W_{out}$ in the training process. For this reason, ESNs have traditionally been widely used for recurrent neural networks which overcome the vanishing gradient problem. For forecasting of a dynamical system, the ground truth output at current timestep is the input at the next timestep, i.e., $y[n] = u[n+1]$, during the training process. During the inference process, the mode output $\hat{y}[n]$ is feed back to the model as input, i.e., $\hat{y}[n] = u[n+1]$.

## Training with Regularization

In Step 6 of previous section, $W_{out}$ is trained to minimize the mean squared error $E_d$ between the ESN predictions and the data [2]:

$E_d = \frac{1}{n_{out}}\sum_{i=1}^{n_{out}}\frac{1}{m}\sum_{n=1}^{m}(\hat{y_i}[n]-y_i[n])^2$

A regularization term can be added to the training process such that

$W_{out} = (YX')(XX' + \lambda I)^{-1}$

where $\lambda$ is a Tikhonov regularization factor. The optimization is

$E_d = \frac{1}{n_{out}}\sum_{i=1}^{n_{out}}\frac{1}{m}\sum_{n=1}^{m}((\hat{y_i}[n]-y_i[n])^2 + \lambda \Vert w_{out,i} \Vert)^2$

where $w_{out,i}$ demotes the i-th row of $W_{out}$. This optimization penalize large values of $W_{out}$, which generally avoids overfitting [2].

## Physics Informed Echo State Network

Assuming the ODE can be written as

$\mathcal{F}(y) \equiv \dot{y} + \mathcal{N}(y)$

where $\mathcal{F}$ is a general non-linear operator, $\dot{}$ is the time derivative and $\mathcal{N}$ is a nonlinear differential operator. Then we can add a physical loss in the training process

$E_{tot} = E_d + E_p$

where $E_p = \frac{1}{n_{out}}\sum_{i=1}^{n_{out}}\frac{1}{p}\sum_{n=1}^{p}|\mathcal{F}(\hat{y_i}[n]|$, here the set $\hat{y_i}[n]$ (n=1, ...., p) denotes the "collocation points". Practically, the optimization of $W_{out}$ is performed using the L-BFGS-B algorithm with the $W_{out}$ obtained in Step 6 by ridge regression as the initial guess.

## Continuous Time Echo State Network

TBC

## References
1.  
Kudithipudi, Dhireesha
Saleh, Qutaiba
Merkel, Cory
Thesing, James
Wysocki, Bryant (2016).
Design and Analysis of a Neuromemristive Reservoir Computing Architecture for Biosignal Processing, Frontiers in Neuroscience, 9, 1-17. [https://doi.org/10.3389/fnins.2015.00502](https://doi.org/10.3389/fnins.2015.00502)

2.
Doan, N. A.K.
Polifke, W.
Magri, L (2020).
Physics-informed echo state networks, Preprint submitted to Journal of Computational Science.


Enter text in [Markdown](http://daringfireball.net/projects/markdown/). Use the toolbar above, or click the **?** button for formatting help.
