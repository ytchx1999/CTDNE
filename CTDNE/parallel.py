import random
import numpy as np
from tqdm import tqdm


def parallel_generate_walks(d_graph, global_walk_length, num_walks, cpu_num, sampling_strategy=None,
                            num_walks_key=None, walk_length_key=None, neighbors_key=None, neighbors_time_key=None,
                            probabilities_key=None,
                            first_travel_key=None, quiet=False):
    """
    Generates the random walks which will be used as the skip-gram input.
    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]
            last_time = -np.inf

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:

                time_mask = []
                walk_options = d_graph[walk[-1]].get(neighbors_key, None)
                if walk_options:
                    for neighbor in walk_options:
                        neighbor_times = d_graph[walk[-1]][neighbors_time_key][neighbor]
                        relevant_neighbor_times = [neighbor_time for neighbor_time in neighbor_times if
                                                   neighbor_time > last_time]
                        if len(relevant_neighbor_times) > 0:
                            time_mask.append(np.min(relevant_neighbor_times))
                        else:
                            time_mask.append(-np.inf)
                time_mask = np.array(time_mask)

                # Skip dead end nodes
                if not walk_options or not np.any(time_mask != -np.inf):
                    break

                min_time = np.min([time for time in time_mask if time != -np.inf])

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key]
                    time_probabilities = np.exp(probabilities * (time_mask - min_time)/60/60/24) / np.sum(
                        np.exp(probabilities * (time_mask - min_time)/60/60/24))
                    walk_to = np.random.choice(walk_options, size=1, p=time_probabilities)[0]
                else:
                    probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
                    time_probabilities = np.exp(probabilities * (time_mask - min_time)/60/60/24) / np.sum(
                        np.exp(probabilities * (time_mask - min_time)/60/60/24))
                    walk_to = np.random.choice(walk_options, size=1, p=time_probabilities)[0]

                last_time = np.min([next_time for next_time in d_graph[walk[-1]][neighbors_time_key][walk_to] if
                                next_time > last_time])

                walk.append(walk_to)

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks
