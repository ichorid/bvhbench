import random
from pprint import pprint

set_size = 100

dims = (0, 1, 2)


def gen_random_point(a=0.0, b=1.0):
    return tuple(random.uniform(a, b) for _ in dims)


testset = tuple(gen_random_point() for _ in xrange(set_size))
enumed = enumerate(testset)


def voxelize_to_point_tuples_tree(pointset, depth=0, max_leaf_size=1):
    d = (depth + 1) % len(dims)
    if len(pointset) <= max_leaf_size:
        return pointset

    testset_sorted = sorted(pointset, key=lambda g: g[d])
    splitting_point = len(testset_sorted) / 2

    return voxelize(testset_sorted[:splitting_point], depth), \
           voxelize(testset_sorted[splitting_point:], depth)


def voxelize(pointset, depth=-1, max_leaf_size=1):
    d = (depth + 1) % len(dims)
    if len(pointset) <= max_leaf_size:
        return pointset

    testset_sorted = sorted(pointset, key=lambda g: g[d])
    splitting_point = len(testset_sorted) / 2

    return testset_sorted[splitting_point][d], voxelize(testset_sorted[:splitting_point], depth + 1), \
           voxelize(testset_sorted[splitting_point:], depth + 1)


def dist(p1, p2):
    dist = 0.0
    for d in [0, 1, 2]:
        dist = dist + (p1[d] - p2[d]) ** 2
    return dist


def find_node_by_point_tree(tree, point, depth=-1):
    d = (depth + 1) % len(dims)
    if isinstance(tree[0], tuple):
        return tree[0]

    print d, tree[0]
    search_left = point[d] < tree[0]
    best = find_node_by_point_tree(tree[1 if search_left else 2], point, depth + 1)
    # There is a chance that there is a better fitting point on the other side of the hyperplane
    if (tree[0] - point[d]) ** 2 < dist(best, point):
        cand = find_node_by_point_tree(tree[1 if not search_left else 2], point, depth + 1)
        best = cand if dist(cand, point) < dist(best, point) else best
    return best


def find_node_by_point_linear(tset, point):
    best_match = tset[0]
    best_match_dist = 0.0
    for p in tset:
        dist_cand = dist(point, p)
        if dist_cand < best_match_dist or best_match_dist == 0.0:
            best_match_dist = dist_cand
            best_match = p
    return best_match


def main():
    tree = voxelize(testset)
    pprint(tree)

    test_point = 0.1, 0.1, 0.4

    found_point_tree = find_node_by_point_tree(tree, test_point)
    print  found_point_tree, dist(found_point_tree, test_point)

    found_point_linear = find_node_by_point_linear(testset, test_point)
    print  found_point_linear, dist(found_point_linear, test_point)


if __name__ == "__main__":
    main()
