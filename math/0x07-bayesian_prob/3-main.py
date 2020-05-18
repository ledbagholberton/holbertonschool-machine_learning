#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    posterior = __import__('3-posterior').posterior

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    a = posterior(26, 130, P, Pr)
    print(a)
    plt.plot(a)
    plt.show()
