---
published: true
---
## What is Echo State Network?

Echo State Network (ESN) is a class of reservoir computing and is considered as partially-trained neural networks. Three main components of ESN are: input, reservoir, and output, as shown in figure below [1].Unlike conventional neural networks, the input weight matrix $W_{in}$ and reservoir layer weight matrix $W_{x}$ are randomly generated and never changed during training or testing phases of the network. The output layer linearly combines the desired output signal from the high-dimensional hidden layer, and its weights are trained during training process.

![echo_state_network]({{ site.baseurl }}/images/Echo_State_Network_2021_08_30.png)

## How ESN is trained
Suppose we have $m$ training vector pairs, where $u[n]$ is the n-th input vector with a size of $(n_{in} \times 1)$ and $y[n]$ is the corresponding output vector with a size of $(n_{out} \times 1)$. Then the input weight matrix $W_{in}$ would have a dimension of $(n_r \times n_{in})$, the reservoir weight matrix $W_x$ has a size of $n_r \times n_r$, and the output matrix $W_{out}$ would have a dimension of $(n_{out} \times n_r)$.

The training steps of ESN are:

1. At initialization, randomly generate the weights $W_{in}$ and $W_x$ for the input and reservoir layers, respectively.

2. Feed the next input vector $u[n+1]$ to the input layer

3. Calculate the response of the reservoir layer using 

$x[n+1] = f^{res} (W_{in} u[n+1] + W_x x[n])$

where $f^{res}$ is the reservoir layer's activation function. Lets check the dimension of both sides of the equation

$(n_r \times 1) = f^{res}((n_r \times n_{in})\times(n_{in} \times 1) + (n_r \times n_r)(n_r \times 1))$ [$\checkmark$]

4. Save the response $x[n+1]$ in a matrix $X$.

5. Repeat steps 2 - 4 for all $m$ training pairs

6. Calculate output weights $W_{out}$ based on the equation below

$W_{out} = (YX')(XX')^{-1}$

Lets check the dimensions of both sides of the above equation

$(n_{out} \times 1) = \left((n_{out}\times m)(m \times n_{in})\right)\left((n_r \times m)(m \times n_r)\right)^{-1} = (n_{out} \times n_r)$ [$\checkmark$]

## References
1.  
Kudithipudi, Dhireesha
Saleh, Qutaiba
Merkel, Cory
Thesing, James
Wysocki, Bryant (2016). 
Design and Analysis of a Neuromemristive Reservoir Computing Architecture for Biosignal Processing, Frontiers in Neuroscience, 9, 1-17. [https://doi.org/10.3389/fnins.2015.00502](https://doi.org/10.3389/fnins.2015.00502)


Enter text in [Markdown](http://daringfireball.net/projects/markdown/). Use the toolbar above, or click the **?** button for formatting help.
