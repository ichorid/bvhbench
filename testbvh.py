import random
from pprint import pprint

set_size = 10

dims = (0, 1, 2)


def gen_random_point(a=0.0, b=1.0):
    return tuple(random.uniform(a, b) for _ in dims)


testset = tuple(gen_random_point() for _ in xrange(set_size))
enumed = enumerate(testset)


def voxelize(pointset, depth=0, max_leaf_size=1):
    d = (depth + 1) % len(dims)
    if len(pointset) <= max_leaf_size:
        return pointset

    testset_sorted = sorted(pointset, key=lambda g: g[d])
    splitting_point = len(testset_sorted) / 2

    return voxelize(testset_sorted[:splitting_point], depth), \
           voxelize(testset_sorted[splitting_point:], depth)

def main():
    pprint(voxelize(testset))


if __name__ == "__main__":
    main()
