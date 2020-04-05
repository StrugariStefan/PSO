import numpy as np
from math import pow

def de_jong(x: np.array) -> float :
    n = len(x)

    for i in range(n) :
        if x[i] < -5.12 or x[i] > 5.12 :
            return np.inf

    return np.sum(x ** 2)

def griewangk(x: np.array) -> float :
    n = len(x)

    for i in range(n) :
        if x[i] < -600 or x[i] > 600 :
            return np.inf

    product = np.prod(np.cos(np.divide(x, np.sqrt(np.arange(n) + 1))))
    return np.sum(x ** 2) / 4000 - product + 1

def rastrigin(x: np.array) -> float :
    n = len(x)

    for i in range(n) :
        if x[i] < -5.12 or x[i] > 5.12 :
            return np.inf

    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x: np.array) -> float :
    n = len(x)
    for i in range(n) :
        if x[i] < -2.048 or x[i] > 2.048 :
            return np.inf

    return np.sum(100 * (x[1:] - x[:n - 1] ** 2) ** 2 + (1 - x[:n - 1]) ** 2)

def six_hump_camel_back(x: np.array) -> float :
    n = len(x)
    if n != 2 :
        raise Exception("Invalid six hump cammel back input size")

    if x[0] < -3 or x[0] > 3 :
        return np.inf

    if x[1] < -2 or x[1] > 2 :
        return np.inf

    return (4 - 2.1 * x[0] ** 2 + pow(x[0], 4) / 3) * x[0] ** 2 + x[0] * x[1] + ( -4 + 4 * x[1] ** 2) * x[1] ** 2


DE_JONG = 'De Jong'
RASTRIGIN = 'Rastrigin'
ROSENBROCK = 'Rosenbrock'
GRIEWANGK = 'Griewangk'
SIX_HUMP_CAMEL_BACK = 'Six hump camel back'

FITNESS = 'fitness'
DOMAIN = 'domain'
OPTIMX = 'optimx'
OPTIMY = 'optimy'

fitness_functions = {
    DE_JONG: {
        FITNESS: de_jong,
        DOMAIN: (-5.12, 5.12), 
        OPTIMX: 0,
        OPTIMY: 0
    },
    RASTRIGIN: {
        FITNESS: rastrigin,
        DOMAIN: (-5.12, 5.12), 
        OPTIMX: 0,
        OPTIMY: 0
    },
    ROSENBROCK: {
        FITNESS: rosenbrock,
        DOMAIN: (-2.048, 2.048), 
        OPTIMX: 1,
        OPTIMY: 0
    },
    GRIEWANGK: {
        FITNESS: griewangk,
        DOMAIN: (-600, 600), 
        OPTIMX: 0,
        OPTIMY: 0
    },
    SIX_HUMP_CAMEL_BACK: {
        FITNESS: six_hump_camel_back,
        DOMAIN: [[-3, 3], [-2, 2]],
        OPTIMX: [[-0.0898,0.7126], [0.0898,-0.7126]],
        OPTIMY: -1.0316
    }
}


if __name__ == '__main__' :
    # test_input = np.zeros(shape = 5)
    test_input = np.random.randint(low = 0, high = 10, size = 2)
    print (test_input)
    
    # dj_result = de_jong(test_input)
    # gr_result = griewangk(test_input)
    # ra_result = rastrigin(test_input)
    # ro_result = rosenbrock(test_input)
    # sh_result = six_hump_cammel_back(test_input)

    print (six_hump_camel_back(np.array([-0.0898,0.7126])))

    # print (dj_result)
    # print (gr_result)
    # print (ra_result)
    # print (ro_result)
    # print (sh_result)