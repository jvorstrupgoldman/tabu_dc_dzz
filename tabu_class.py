import numpy as np
import sys
from scipy.spatial.distance import hamming
from time import time


def progressBar(value, endvalue, bar_length=40):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()


def tabu_sampler(dim,  # Dimension of the target
                no_generators,  # Number of generators
                energy_function,  # Calculates the energy of the target density
                calculate_rates,  # Calculates the initial rates for the problem
                update_rates,  # Updates the rate after a flip has been carried out
                apply_generator,  # Applies the generator to the state
                current_state, baseline_state, target_time, thin_rate, time_change_rate,
                calc_energy=True, calc_hamming=True, symmetry=True, SAW_bool=True,
                print_output=False, save_samples=True, print_rate=100):
    """ Runs the Tabu sampler.
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
            time_change_rate: Only applicable if SAW=True (SAW is short-hand for self-avoiding walk); how often to flip tau
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
                Outputs:
                    Jump rates: locally-balanced rates for each generator
            update_rates:
                Inputs:
                    current_state: current value of Markov process
                    jump_rates: current jump rates
                    jump_rates_sum: sum of jump rates
                    generator_index: the generator that was chosen
                Outputs:
                    Jump rates post-generator application, for entire problem
            apply_generator:
                Inputs:
                    current_state: current value of Markov process
                    generator_index: the generator that was chosen
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
        The length of the average Excursion
        Average jump time of the sampler

    """
    # Initialisation
    # Problem parameters
    iteration = 0
    refreshment_iteration = 0
    t = 0
    sample_index = 1
    length_of_saw = 0

    if SAW_bool==False:
        time_change_rate = 10e-20

    # Placeholders
    samples = np.ones([int(target_time/thin_rate), dim])
    hamming_distances = np.zeros(int(target_time/thin_rate))
    energy = np.zeros(int(target_time/thin_rate))
    alpha = np.ones(no_generators)
    
    # If symmetric generators, build an index matrix
    if symmetry:
        sym_matrix = np.arange(0, dim**2).reshape(dim, dim)

    # Insert initial values
    samples[0, :] = current_state
    energy[0] = energy_function(current_state)
    hamming_distances[0] = dim*hamming(current_state, baseline_state)

    # Calculate all rates initially and return the normalising constant
    jump_rates = calculate_rates(current_state)
    jump_rates_sum = (jump_rates*alpha).sum()
    jump_rates_sum_alpha_flip = (jump_rates*(np.ones(no_generators) - alpha)).sum()

    start_time = time()
    while t < target_time:
        # Initially we verify that we our current SAW is doing the right thing of moving forward
        # If yes, carry out the SAW algorithm
        if jump_rates_sum > jump_rates_sum_alpha_flip:
            if jump_rates_sum > 0:
                jump_time = np.random.exponential(size=1, scale=float(jump_rates_sum ** -1))
            else:
                # Create unrealistically large jump to ensure that switch is carried out
                jump_time = 1e10

            # Generate refreshment time
            refreshment_time = np.random.exponential(size=1, scale=float(time_change_rate**-1))

            # Actual event
            event_time = min(jump_time, refreshment_time)

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

                if calc_hamming:
                    # Calculate Hamming distance
                    hamming_distances[sample_index] = dim*hamming(current_state, baseline_state)
                # Update iterator
                sample_index += 1

            # Check if we are having a refreshment; if yes, update
            if refreshment_time < jump_time:
                if SAW_bool:
                    # Replace currently acceptable directions with the complement
                    alpha = np.ones(no_generators) - alpha

                    # Update jump rates with the complementary set
                    jump_rates_sum = (jump_rates*alpha).sum()

                    if print_output:
                        print("SAW Length:", length_of_saw)

                    length_of_saw = 0
                    refreshment_iteration += 1
                else:
                    current_state = np.random.permutation(dim)

            else:
                # Pick which index to apply action to
                generator_index = np.searchsorted((jump_rates*alpha).cumsum(), np.random.rand(1) * jump_rates_sum)[0]

                # Update the state for that index
                current_state = apply_generator(current_state, generator_index)

                # Calculate the energy for the new (currently not possible) flip and relevant neighbours
                # and return the new sum of jump_rates
                jump_rates = update_rates(current_state, jump_rates, jump_rates_sum, generator_index)

                if SAW_bool:
                    # Remove the flip from the set of possible flips
                    alpha[generator_index] = 0

                    # If the generator is symmetric around an axis, remove the corresponding symmetric value as well
                    if symmetry:
                        # Find values corresponding to the matrix swap
                        i_val = int(np.floor(generator_index/dim))
                        j_val = generator_index % dim
                        # Replace the symmetric value
                        alpha[sym_matrix[j_val, i_val]] = 0

                    # Add one to length of self-avoiding walk
                    length_of_saw += 1

                # Calculates the sum of the valid jump-rates for the choice
                jump_rates_sum = (jump_rates*alpha).sum()
                jump_rates_sum_alpha_flip = (jump_rates * (np.ones(no_generators) - alpha)).sum()

                if print_output:
                    print("Chosen generator:", generator_index)
                    print("Updated state:", current_state)
                    if symmetry:
                        print("New jump rates:\n", (jump_rates*alpha).round(1).reshape(dim, dim))
                    else:
                        print("New jump rates:\n", (jump_rates*alpha).round(1))
                    print("New jump-rate sum:", jump_rates_sum)
                    print("New Alpha:", alpha)

        # If the alternative jump rates are larger,
        # carry out a check to see if Tau should be refreshed
        else:
            if jump_rates_sum_alpha_flip > 0:
                jump_time = np.random.exponential(size=1, scale=float(jump_rates_sum_alpha_flip ** -1))
            else:
                # Create unrealistically large jump to ensure that switch is carried out
                jump_time = 1e10

            # Update runtime
            t += jump_time

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

                if calc_hamming:
                    # Calculate Hamming distance
                    hamming_distances[sample_index] = dim*hamming(current_state, baseline_state)
                # Update iterator
                sample_index += 1

            # Calculate the probability of flipping tau
            flip_probability = (jump_rates_sum_alpha_flip - jump_rates_sum)/jump_rates_sum_alpha_flip

            # Check if it is time to switch to the complement of the generators
            if np.random.rand(1) < flip_probability:
                # Replace currently acceptable directions with the complement
                alpha = np.ones(no_generators) - alpha

                # Update jump rates with the complementary set
                jump_rates_sum = (jump_rates * alpha).sum()
                jump_rates_sum_alpha_flip = (jump_rates*(np.ones(no_generators) - alpha)).sum()

                if print_output:
                    print("Self-avoiding Walk Length:", length_of_saw)

                length_of_saw = 0
                refreshment_iteration += 1
            else:
                # Pick which index to apply action to
                generator_index = np.searchsorted((jump_rates*alpha).cumsum(), np.random.rand(1) * jump_rates_sum)[0]

                # Update the state for that index
                current_state = apply_generator(current_state, generator_index)

                # Calculate the energy for the new (currently not possible) flip and relevant neighbours
                # and return the new sum of jump_rates
                jump_rates = update_rates(current_state, jump_rates, jump_rates_sum, generator_index)
                jump_rates_sum_alpha_flip = (jump_rates * (np.ones(no_generators) - alpha)).sum()

                if SAW_bool:
                    # Remove the flip from the set of possible flips
                    alpha[generator_index] = 0

                    # If the generator is symmetric around an axis, remove the corresponding symmetric value as well
                    if symmetry:
                        # Find values corresponding to the matrix swap
                        i_val = int(np.floor(generator_index/dim))
                        j_val = generator_index % dim
                        # Replace the symmetric value
                        alpha[sym_matrix[j_val, i_val]] = 0

                    # Add one to length of self-avoiding walk
                    length_of_saw += 1

                # Calculates the sum of the valid jump-rates for the choice
                jump_rates_sum = (jump_rates*alpha).sum()

                if print_output:
                    print("Chosen generator:", generator_index)
                    print("Updated state:", current_state)
                    if symmetry:
                        print("New jump rates:\n", (jump_rates*alpha).round(1).reshape(dim, dim))
                    else:
                        print("New jump rates:\n", (jump_rates*alpha).round(1))
                    print("New jump-rate sum:", jump_rates_sum)
                    print("New Alpha:", alpha)

        # Update iteration counter
        iteration += 1

        if iteration % print_rate == 0:
            progressBar(t, target_time)
            
    runtime = time() - start_time
            
    if SAW_bool:
        print("Average Excursion:", np.floor(iteration/refreshment_iteration))
    print("Average jump length:", np.round(t/iteration, 8))
    print("Runtime:", np.round(runtime, 2))
    
    return samples, energy, hamming_distances, alpha, iteration, runtime
