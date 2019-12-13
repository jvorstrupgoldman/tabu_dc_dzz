import numpy as np
import sys
from scipy.spatial.distance import hamming
from time import time
from decimal import *

def progressBar(value, endvalue, bar_length=40):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()


def hz_sampler(dim,  # Dimension of the target
                no_generators,  # Number of generators
                energy_function,  # Calculates the energy of the target density
                calculate_rates,  # Calculates the initial rates for the problem
                update_rates,  # Updates the rate after a flip has been carried out
                apply_generator,  # Applies the generator to the state
                current_state, target_time, thin_rate,
                calc_energy=True, print_rate=100):
    """ Runs the Self-avoiding Random Walk sampler.
    Note that the locally-balancing function must be defined outside of the sampler;
    the user will define it when defining the rate functions.

    Inputs:
    -------
        Parameters:
        -----------
            dim: Cardinality of state-space
            no_generators: The number of generators for the space
            current_state: Initial value
            baseline_state: reference state to calculate Hamming distance (can be mode, mean, etc.)
            target_time: How long in cts. time the algorithm should run for
            thin_rate: How often the samples should be thinned
            time_change_rate: Only applicable if SAW; how often to flip tau
            calc_energy: If the energy should be calculated at each iteration
            symmetry: If the generator set is symmetric around a diagonal
            print_output: For debugging
            save_samples: bool for actually saving samples; useful for high-dimensional problems

        Functions:
        ----------
            energy_function:
                Inputs:
                    current_state: current value of Markov process
                Outputs:
                    Energy of the target distribution
            calculate_rates:
                Inputs:
                    current_state: current value of Markov process
                    theta: current direction of generators
                Outputs:
                    Jump rates forward: locally-balanced rates for each generator
                    Jump rates backwards: locally-balanced rates for each inverse generator
            update_rates:
                Inputs:
                    current_state: current value of Markov process
                    theta: current direction of generators
                    jump_rates: current jump rates
                    generator_index: the generator that was chosenxtg
                Outputs:
                    Jump rates forward: locally-balanced rates for each generator after gen i was applied
                    Jump rates backwards: locally-balanced rates for each inverse generator after gen i was applied
            apply_generator:
                Inputs:
                    current_state: current value of Markov process
                    generator_index: the generator that was chosen
                    theta: current direction of generators
                Outputs:
                    Current state after application of generator

    Outputs:
    --------
        Samples at each thinned time-point
        Energy at each thinned time-point
        Hamming distance relative to the baseline state
        Active jumps available
        Number of events

    Prints:
    -------
        The length of the average SAW
        Average jump time of the sampler

    """
    # Initialisation
    # Problem parameters
    iteration = 0
    t = 0
    sample_index = 1

    # Placeholders
    samples = np.ones([int(target_time/thin_rate), dim])
    energy = np.zeros(int(target_time/thin_rate))

    # Insert initial values
    samples[0, :] = current_state
    energy[0] = energy_function(current_state)

    jump_rates = calculate_rates(current_state)

    jump_rates_sum = jump_rates.sum()

    start_time = time()
    while t < target_time:
        # Calculate all rates initially and return the normalising constant

        # If yes, carry out the SAW algorithm as initially developed
        event_time = np.random.exponential(size=1, scale=float(jump_rates_sum ** -1))

        # Update runtime
        t += event_time

        # While thinner
        while t > sample_index * thin_rate:
            # If all samples have been collected, break
            if sample_index >= int(target_time/thin_rate):
                print(t)
                break
            # Insert sample
            samples[sample_index, :] = current_state
            if calc_energy:
                # Insert current energy
                energy[sample_index] = energy_function(current_state)

            # Update iterator
            sample_index += 1

        # Pick which index to apply action to
        generator_index = np.searchsorted(jump_rates.cumsum(), Decimal(np.random.rand(1)[0]) * Decimal(jump_rates_sum))

        # Carry out update to the chosen generator
        current_state = apply_generator(current_state, generator_index)

        jump_rates = calculate_rates(current_state)

        jump_rates_sum = jump_rates.sum()

        # Update iteration counter
        iteration += 1

        if iteration % print_rate == 0:
            progressBar(t, target_time)
            
    runtime = time() - start_time

    print("Average jump length:", np.round(t/iteration, 8))
    print("Runtime:", np.round(runtime, 2))
    
    return samples, energy, iteration, runtime
