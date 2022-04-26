FORMAT = '%(levelname)s — %(asctime)s — %(message)s: Iteration: %(iteration)s — Best parameters: %(best_parameters)s — Best parameters fitness: %(best_parameters_fitness)s — Duration: %(duration)s'


def format_log(iteration, best_metric, best_specification, duration):
    return {'iteration': iteration, 'best_parameters_fitness': best_metric, 'best_parameters': best_specification, 'duration': duration}
