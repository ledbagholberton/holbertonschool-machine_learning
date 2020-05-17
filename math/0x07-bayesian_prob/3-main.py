#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    posterior = __import__('3-posterior').posterior

    P = np.linspace(0, 1, 101)
    Pr = np.ones(101) / 101
    print(posterior(26, 130, P, Pr))
