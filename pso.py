from objectives import rastrigin, rosenbrock, griewangk, six_hump_camel_back, de_jong

import numpy as np
from math import sqrt
from typing import Callable

def get_best_known_position(x: np.array, objective: Callable[[np.array], float], current: np.array = np.array([])) -> np.array :
    g_index = np.argmin(np.apply_along_axis(objective, axis = 1, arr = x))
    g = x[g_index].copy()

    if len(current) != 0 and objective(current) < objective(g):
        g = current

    return g

def get_best_known_for_each_particle(x: np.array, p: np.array, objective: Callable[[np.array], float]) -> np.array :
    size = x.shape[0]
    p_copy = p.copy()

    for i in range(size) :
        if objective(x[i]) < objective(p[i]) :
            p_copy[i] = x[i]

    return p_copy

def init(domain: np.array, objective: Callable[[np.array], float],  population_size: int = 5) -> tuple :
    if domain.shape[1] != 2 :
        raise Exception("Invalid domain")

    n = domain.shape[0]
    x = np.random.uniform(low = domain[:, 0], high = domain[:, 1], size = (population_size, n))

    domain_range = domain[:, 1] - domain[:, 0]
    v = np.random.uniform(low = - domain_range, high = domain_range, size = (population_size, n))

    p = x.copy()

    return x, v, p


def pso(fitness: Callable[[np.array], float], domain: np.array, w1: float = 0.2, w2: float = 0.2, w3: float = 0.2, population_size: int = 100, T_MAX = 100) :
    x, v, p = init(domain, fitness, population_size = population_size)

    g_list = []
    g = np.array([])

    t = 0
    while t < T_MAX :
        t += 1

        g = get_best_known_position(x, fitness, g)
        g_list.append(g)
        p = get_best_known_for_each_particle(x, p, fitness)

        for i in range(population_size) :
            v[i] = w1 * v[i] + w2 * np.random.uniform() * (p[i] - x[i]) + w3 * np.random.uniform() * (g - x[i])
            x[i] += v[i]

    return g, g_list


def get_domain(interval: tuple, dimensions_number: int = 2) -> np.array :
    low = interval[0]
    high = interval[1]

    if low >= high :
        raise Exception('Invalid interval')
    return np.array([[low, high]] * dimensions_number)

if __name__ == '__main__' :
    # dj_domain = np.array([[-5.12, 5.12], [-5.12, 5.12]])
    # ra_domain = np.array([[-5.12, 5.12], [-5.12, 5.12]])
    # sh_domain = np.array([[-3, 3], [-2, 2]])
    # ro_domain = np.array([[-2.048, 2.048], [-2.048, 2.048]])
    # gr_domain = np.array([[-600, 600], [-600, 600]])

    # pso(de_jong, dj_domain)
    # pso(rastrigin, ra_domain)
    # pso(six_hump_camel_back, sh_domain)
    # pso(rosenbrock, ro_domain)
    # pso(griewangk, gr_domain)

    domains = {
        rastrigin: (-5.12, 5.12),
        de_jong: (-5.12, 5.12)
    }

    fitness_functions = [de_jong, rastrigin]
    dims = [2, 5, 10, 30]

    for ff in fitness_functions :
        for d in dims :
            dom = get_domain(domains[ff], d)
            g_list = pso(ff, dom)

            print (ff(g_list[-1]))