def scrambled_halton_sequence_generator(dimension=1, permutation=None):

    primes = prime_generator()
    bases = [primes.next() for _ in xrange(dimension)]
    if permutation is None:
        generators = [halton_sequence_generator(base) for base in bases]
    elif permutation == 'reverse':
        generators = [halton_sequence_generator(base, permutation=[0] + [base-i for i in xrange(1,base)]) for base in bases]
    elif permutation == 'Braaten-Weller':
        assert dimension<=16, "Braaten-Weller only supports dimensions up to 16"
        perms ={2: [0, 1],
                3: [0, 2, 1],
                5: [0, 2, 4, 1, 3],
                7: [0, 3, 5, 1, 6, 2, 4],
                11: [0 ,5 ,8 ,2 ,10 ,3, 6, 1, 9, 4, 7],
                13: [0, 6, 10, 2, 8, 4, 12, 1, 9, 5, 11, 3, 7],
                17: [0, 8, 13, 3, 11, 5, 16, 1, 10, 7, 14, 4, 12, 2, 15, 6, 9],
                19: [0, 9, 14, 3, 17, 6, 11, 1, 15, 7, 12, 4, 18, 8, 2, 16, 10, 5, 13],
                23: [0, 11, 17, 4, 20, 7, 13, 2, 22, 9, 15, 5, 18, 1, 14, 10, 21, 6, 16, 3, 19, 8, 12],
                29: [0, 14, 22, 5, 18, 9, 27, 2, 20, 11, 25, 7, 16, 3, 24, 13, 19, 6, 28, 10, 1, 23, 15, 12, 26, 4, 17, 8, 21],
                31: [0, 16, 8, 26, 4, 22, 13, 29, 2, 19, 11, 24, 6, 20, 14, 28, 1, 17, 9, 30, 10, 23, 5, 21, 15, 3, 27, 12, 25, 7, 18],
                37: [0, 18, 28, 6, 23, 11, 34, 3, 25, 14, 31, 8, 20, 36, 1, 16, 27, 10, 22, 13, 32, 4, 29, 17, 7, 35, 19, 2, 26, 12, 30, 9, 24, 15, 33, 5, 21],
                41: [0, 20, 31, 7, 26, 12, 38, 3, 23, 34, 14, 17, 29, 5, 40, 10, 24, 1, 35, 18, 28, 9, 33, 15, 21, 4, 37, 13, 30, 8, 39, 19, 25, 2, 32, 11, 22, 36, 6, 27, 16],
                43: [0, 21, 32, 7, 38, 13, 25, 3, 35, 17, 28, 10, 41, 5, 23, 30, 15, 37, 1, 19, 33, 11, 26, 42, 8, 18, 29, 4, 39, 14, 22, 34, 6, 24, 12, 40, 2, 31, 20, 16, 36, 9, 27],
                47: [0, 23, 35, 8, 41, 14, 27, 3, 44, 18, 31, 11, 37, 5, 25, 39, 16, 21, 33, 1, 46, 12, 29, 19, 42, 7, 28, 10, 36, 22, 4, 43, 17, 32, 13, 38, 2, 26, 45, 15, 30, 6, 34, 20, 40, 9, 24],
                53: [0, 26, 40, 9, 33, 16, 49, 4, 36, 21, 45, 12, 29, 6, 51, 23, 38, 14, 43, 1, 30, 19, 47, 10, 34, 24, 42, 3, 27, 52, 15, 18, 39, 7, 46, 22, 32, 5, 48, 13, 35, 25, 8, 44, 31, 17, 50, 2, 37, 20, 28, 11, 41],
        }
        generators = [halton_sequence_generator(base, permutation=perms[base]) for base in bases]
    elif permutation == 'random':
        generators = [halton_sequence_generator(base, permutation='pseudo-random') for base in bases]
    else:
        print "Permutation %s not supported"%permutation
    while True:
        yield [gen.next() for gen in generators]

quasi_random_number_generator = scrambled_halton_sequence_generator

def prime_generator():
    D = {}
    q = 2
    while True:
        if q not in D:
            yield q
            D[q * q] = [q]
        else:
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]
        q += 1

def vdc(n, q, base, permutation):
    vdc, denom = 0,1
    while n:
        denom *= base
        q /= base
        n, remainder = divmod(n, base)
        vdc += permutation[remainder] * q
    return vdc

def halton_sequence_generator(base, permutation=None, rational=False):
    if permutation is None:
        permutation = range(base) #no permutation
    if permutation == 'pseudo-random':
        import random

    denominator = base
    while True:
        for i in xrange(denominator/base,denominator):
            if permutation == 'pseudo-random':
                l = range(1,base)
                random.shuffle(l)
                permutation = [0] + l

            numerator = vdc(i, denominator, base, permutation)
            #print numerator,denominator

            if rational:
                yield numerator,denominator
            else:
                yield numerator / float(denominator)
        denominator *= base

""" for comparison """
def pseudo_random_sequence_generator(dimension=1):
    import random
    while True:
        yield [random.random() for _ in xrange(dimension)]


if __name__=="__main__":

    """Plot the worst dimensions of the quasi-random number generators, together with a pseudo-random number generator"""

    import numpy as np
    import matplotlib.pyplot as plt
    d = 3
    points = 100
    plt.figure('Normal quasi-random')
    generator = scrambled_halton_sequence_generator(dimension = d)
    samples = np.array([generator.next() for _ in xrange(points)])
    plt.scatter(samples[:,-2],samples[:,-1],marker=',',edgecolor='none',s=1)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.figure('quasi-random with reverse permutation')
    generator = scrambled_halton_sequence_generator(dimension = d, permutation='reverse')
    samples = np.array([generator.next() for _ in xrange(points)])
    #print samples[0,:]
    plt.scatter(samples[:,-2],samples[:,-1],marker=',',edgecolor='none',s=1)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.figure('quasi-random with Braaten-weller permutation')
    generator = scrambled_halton_sequence_generator(dimension = d, permutation='Braaten-Weller')
    samples = np.array([generator.next() for _ in xrange(points)])
    plt.scatter(samples[:,-2],samples[:,-1],marker=',',edgecolor='none',s=1)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.figure('quasi-random with pseudo-random permutation')
    generator = scrambled_halton_sequence_generator(dimension = d, permutation='random')
    samples = np.array([generator.next() for _ in xrange(points)])
    plt.scatter(samples[:,-2],samples[:,-1],marker=',',edgecolor='none',s=1)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.figure('actual pseudo-random')
    generator = pseudo_random_sequence_generator(dimension = d)
    samples = np.array([generator.next() for _ in xrange(points)])
    plt.scatter(samples[:,-2],samples[:,-1],marker=',',edgecolor='none',s=1)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()


