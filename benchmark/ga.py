import time

import pygad

from utils import *


def two_points_crossover(parents):
    parent_1, parent_2 = np.array(parents)
    idx_point_1, idx_point_2 = np.sort(np.random.permutation(np.arange(1, parents.shape[1]))[:2])
    children = np.concatenate((parent_1[:idx_point_1], parent_2[idx_point_1:idx_point_2],
                               parent_1[idx_point_2:]))
    return children


def crossover_func(parents, offspring_size, ga_instance):
    # Get current population
    idx_parents = ga_instance.last_generation_parents_indices
    current_parents_fitness = ga_instance.last_generation_fitness[idx_parents]
    # Create a new offspring
    offspring = np.empty(parents.shape)
    for i in range(len(parents)):
        # Pick two random parents
        idx_parents_cross = np.random.choice(len(parents), 2, replace=False)
        parents_cross = parents[idx_parents_cross]
        # According to a crossover probabilty
        if np.random.random() <= ga_instance.crossover_probability:
            # Apply two-points crossover
            offspring[i] = two_points_crossover(parents_cross)
        else:
            # Replace with the best parent
            offspring[i] = parents[idx_parents_cross[np.argmin(current_parents_fitness[
                                                                   idx_parents_cross])]]
    return offspring.astype(int)


def on_generation(ga_instance):
    # Set the new mutation rate
    current_mutation = ga_instance.mutation_probability
    current_iteration = ga_instance.generations_completed
    mutation_rate = current_mutation - current_mutation * (current_iteration // 5) * 0.3
    if mutation_rate < 0.01:
        mutation_rate = 0.01
    ga_instance.mutation_probability = mutation_rate


def on_fitness(ga_instance, solutions_fitness):
    solution_fitness = np.max(solutions_fitness)
    solution = ga_instance.best_solutions[-1]
    d = format_log(ga_instance.generations_completed + 1, run, total_run, solution_fitness,
                   solution, None)
    logging.warning("Current best solution", extra=d)


def run_ga(run):
    # Stop criteria: patience value or number of generations
    stop_criteria = 'saturate_10'
    num_generations = 50
    # Population
    num_genes = 14
    sol_per_pop = 15
    init_range_low = 0
    init_range_high = 2
    gene_type = int
    # Elitism
    keep_parents = 2
    # Crossover
    crossover_probability = 0.9
    parent_selection_type = "rws"
    num_parents_mating = sol_per_pop - keep_parents
    # Mutation
    mutation_type = "random"
    mutation_start_probability = 0.2
    # Fitness function
    fitness_function = fitness_func
    # Create instance of GA
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_type=gene_type,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_func,
                           mutation_type=mutation_type,
                           mutation_probability=mutation_start_probability,
                           crossover_probability=crossover_probability,
                           on_generation=on_generation,
                           stop_criteria=stop_criteria,
                           on_fitness=on_fitness,
                           save_best_solutions=True
                           )

    # Run GA and get the best solution
    start = time.time()
    try:
        ga_instance.run()
    except Exception as e:
        end = time.time()
        duration = end - start
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        d = format_log(iteration, run, total_run, solution_fitness, solution, duration)
        logging.error("Exception occurred", exc_info=True, extra=d)
    end = time.time()
    duration = end - start
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    return solution, solution_fitness, solution_idx, duration


# Prepare parameters
total_run = 50
iteration = 0
# Load the data
prepare_data()
# Prepare to store the information about the best global solution
solution_best_global = None
solution_fitness_best_global = -np.inf
solution_idx_best_global = None
duration_best_global = None
# Run GA several times
for run in range(1, total_run + 1):
    d = format_log(None, run, total_run, None, None, None)
    logging.info('**** Starting the GA ****', extra=d)
    solution, solution_fitness, solution_idx, duration = run_ga(run)
    d = format_log(None, run, total_run, solution_fitness, solution, duration)
    logging.info('**** Finishing the GA ****', extra=d)
    if solution_fitness > solution_fitness_best_global:
        solution_best_global = solution
        solution_fitness_best_global = solution_fitness
        solution_idx_best_global = solution_idx
        duration_best_global = duration
d = format_log(None, None, None, solution_fitness_best_global, solution_best_global,
               duration_best_global)
logging.info('**** Best solution of the GA ****', extra=d)
