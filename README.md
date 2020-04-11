# Code for the paper 'Accelerated Sampling on Discrete Spaces with Non-Reversible Markov Processes'
Implementation of the Tabu Sampler, discrete Coordinate Sampler and discrete Zig-Zag Sampler; three MCMC samplers introduced in [Accelerated Sampling on Discrete Spaces with Non-Reversible Markov Processes](https://arxiv.org/abs/1912.04681) by Sam Power and myself. 

## Use of the algorithms
To apply any of the algorithms to your own statistical model, you will need to load the corresponding *algoname_class.py* file. The resulting sampler is a function denoted `algoname_sampler`. 
In the associated docstring of each function, a short description of each argument is presented to clarify what you need to specify. Furthermore, each sampler is judiciously commented nearly line for line, so better understanding of the algorithms can be achieved just by going through the code. Of course, our implementation is just one possible solution, and we in no way claim that it is optimal. If you have any suggestions in how to improve the code, please reach out. 

### Implementing a model
There is quite a few inputs, the first few specifies the dimension of the problem and the number and type of generators, if applicable. 

Subsequently, four functions needs to be specified by the user; these are shared for all three algorithms:

1. Energy function: calculates the log-probability of any configuration.
2. Calculation of jump rates: returns the locally balanced rates using your chosen balancing function for every possible generator 
3. Update of jump rates: in many cases, the ratio of discrete probability distributions are much simpler to calculate than the actual probability distribution itself. If this is the case, significant speed-ups can be achieved if this update can be implemented. If not, you can simply pass the calculation function described above here instead.
4. Application of a generator: A simple function that updates the current configuration whenever a generator is applied.

For more details on these functions and how to exactly specify their input and output, please refer to the docstring. The work of applying any of the algorithms is in writing these functions.

A few more parameters then follow and are easily specified. As our main comparison is the continuous-time algorithm of [Zanella](https://arxiv.org/pdf/1711.07424) which is most easily compared with the Tabu sampler, this algorithm can be activated simply by setting `SAW_bool=False` in the `tabu_sampler` function argument.  

## Examples
All 7 examples presented in the paper are included above as [Jupyter notebooks](https://jupyter.org/). The 5 starting with *TABU* are all examples where the generator have order 2, while the two starting with *PDMP* are cases where the generators are of higher orders. As our goal was to present the broad versatility of the samplers via what, we hope, is a wide array of relevant discrete models, we highly recommend users to take a look at the example most relevant to them to see how they can implement their own algorithm. 
