# Hopfield_Nets
This is an implementation of Hopfield network

**What is Hopfield Neural network?**

This is a network of interconnected nodes (neurons), each governed by dynamic equations of specific form. In the simplest formulation (discrete, and symmetric connectivity matrix), the neuron can be only in two states: +1 (up), and -1 (down). The neuron affects all the connected to it neighboring neurons by synapses: positive connections force the two connected neurons be in the same state, and negative connections tend to make neurons be in the opposite states.

<img src='../imgs_for_jupyter/HN.png' width="240" height="240" align="center"/>

**Equations for discrete Hopfiel Neural network of size $N$ (vector form)**:

$$h[n+1] =  Ws[n] + b $$
$$s[n+1] = sgn(h[n+1])$$


where
$\textbf{n}$ represents a number of a step during the evolution of the system

$\textbf{h}$ - is the column vector hidden state (could be regarded as a membrane potential of a neuron, or $\textbf{local field}$),

$\textbf{s}$ - is the column vector of states of neurons, (The variable ${ (\boldsymbol{s_{i}}+1)/2 }$ could be considered as firing rate of $i_{th}$ neuron)

$\textbf{W}$ - is the connectivity matrix $N \times N$, where the synaptic weights are stored, 

$\textbf{b}$ - is the bias (could be considered as a constant current injected into a neuron)

This network is very similar to dynamics of the atomic spins interacting with each other in solid states.
Thus one can define an energy function:

For two neurons, the energy of their interaction is : $$H_{i,j} = -w_{ij}s_i s_j$$

For the whole system of $N$ neurons:

$$ H = - s^{T} W s - (b,s)$$

This energy function has the local extrema - the fixed points of the dynamics. Once the network arrives at this states, it no longer escapes it (if there is no noise). If we set $b = 0$ The positions of these local minima are encoded in the weight matrix $W$. This makes the system ideal for storage of binary vectors (patterns). 

Once the system has learned (the weights has been adjusted accordingly) to store the patterns, one can initialise the system in one of these patterns corrupted with noise, and if the noise level is sufficiently small, the system will relax to the state corresponding to the closest pattern it has memorised.

# Hebbian learning

The simplest rule to learn these patterns by network is: 'The neurons which fire together - wire together' - famous **Hebb's rule**. Lets call the $\textbf{p}$ patterns to memorise by $\boldsymbol{\sigma^{\nu}}$, where $\nu \in \{1,.,p\}$. 

In the context of Hopfield Neural networks the rule will be represented as:
$$w_{i,j} = \frac{1}{N}\sum_{\nu}\boldsymbol{\sigma^{\nu}_{i}}\boldsymbol{\sigma^{\nu}_{j}}$$

or in vectorised form (if we set $Z = \{\boldsymbol{\sigma^1} ,\boldsymbol{\sigma^2} \dots \boldsymbol{\sigma^p}\}$ - a $N \times p$ matrix consisting of collumn vectors of patterns stacked together horizontally):

$$W = \frac{1}{N}ZZ^T$$

Let's call the $\boldsymbol{m^{\nu}} = \frac{1}{N}(\sigma^{\nu},s)$ - an overlap of the current state $s$ with the pattern $\sigma^{\nu}$. If we consider an overlaps between two patterns - $\sigma^{\nu}$ and $\sigma^{\mu}$ - we denote their overlap as $M^{\mu\nu}$, where $M$ is the matrix of overlaps of patterns (**Important note**: $M = \frac{1}{N}Z^T Z$)

Consider $s[n]$ to be equal one of the patterns, say $\sigma^{\nu}$. Then the state on the next time step is:

$$s[n+1] = sgn\Big(\frac{1}{N}ZZ^T s[n]\Big) = sgn\Big(\frac{1}{N}ZZ^T \sigma^{\nu}\Big) = sgn\Big(ZM[:,\nu]\Big) = sgn\Big(\sum_{\mu} M^{\mu\nu}\sigma^{\mu}\Big) = sgn\Big(\sigma^{\nu} + \sum_{\mu \neq \nu} M^{\mu\nu}\sigma^{\mu}\Big)$$

where $M[:,\nu]$ is a size $p$ column vector of overlaps of patterns in $Z$ with $\sigma^{\nu}$. If the overlaps of $\sigma^{\nu}$ and other $\sigma^{\mu}$ are sufficiently small (which is usually the case for random binary vectors and large $N$) then $s[n+1]$ is exactly equal to $\sigma^{\mu}$, which means that the state is a fixed point of the dynamics.

The asymptotic storage capacity of Hopfield Network with hebbian rule is $O\big(\frac{N}{2 \ln N}\big)$

# Pseudoinverse rule 

But Hebbian rule is quite limited in terms of memory capcacity. A better option is **pseudoinverse** rule:

$$W = ZZ^+$$

where $^+$ denotes Moore-Penrouse inversion: $Z^+ = (Z^TZ)^{-1}Z^T$

Consider, once again, $s[n] = \sigma^{\nu}$. Then the state on the next time step is:

$$s[n+1] = sgn\Big(ZZ^+ s[n]\Big) = sgn\Big(Z(Z^TZ)^{-1}Z^T \sigma^{\nu}\Big) = sgn\Big(ZM^{-1}M[:,\nu]\Big) = sgn(\sigma^{\nu})$$

So, if there is no noise in the system, and the network is already in state which is represented by one of the patterns, it stays there. And it is much more stable than Hebbian rule!

The asymptotic storage capacity of Hopfield Network with pseudoinverse learning rule is $O(N)$. But, it is neither local, nor incremental, so it is not biologically plausible, as the network can't learn patterns one by one (non-incremental), and for the weight update, the synapse requires information which is not available at the spot (non-local) 

# Storkey rule
There is yet another rule, which performs pretty well, and at the same time is biologically plausible (it's both incremental and local).

$$w_{ij} = \sum_{\mu}(\sigma^{\mu}_i - f_{ij})(\sigma^{\mu}_j - f_{ji})$$

where $f_{ij}$ is a local field on the neuron $i$ except the contribution of a neuron $j$: 

$$f_{ij} = \sum_{k}w_{ik}s_k - w_{ii}s_i - w_{ij}s_j $$

The asymptotic storage capacity of Hopfield Network with Storkey learning rule is $O\Big(\frac{n}{\sqrt[3]{4 \ln n}}\Big)$

According to Storkey's thesis, this rule is an approximation to a pseudoinverse rule.

Using Neumann series for inverse matrix ( $(I-T)^{-1} = \sum_i T^i$ ), one can approximate $(Z^TZ)^{-1}$ term in pseudoinverse rule (but, of course, there is more to it).



