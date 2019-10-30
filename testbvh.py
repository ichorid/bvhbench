import pickle
import random
from timeit import default_timer as timer

with open("hand_small.pickle", 'rb') as f:
    HAND = pickle.load(f)

set_size = 100000

dims = (0, 1, 2)

random.seed(123)


def gen_random_point(a=-1.0, b=1.0):
    return tuple(random.uniform(a, b) for _ in dims)


# testset = tuple(gen_random_point() for _ in xrange(set_size))
testset = HAND
enumed = enumerate(testset)


def find_node_by_point_linear(tset, point):
    best_match = tset[0]
    best_match_dist = 0.0
    for p in tset:
        dist_cand = dist(point, p)
        if dist_cand < best_match_dist or best_match_dist == 0.0:
            best_match_dist = dist_cand
            best_match = p
    return best_match


def voxelize_to_point_tuples_tree(pointset, depth=0, max_leaf_size=1):
    d = (depth + 1) % len(dims)
    if len(pointset) <= max_leaf_size:
        return pointset

    testset_sorted = sorted(pointset, key=lambda g: g[d])
    splitting_point = len(testset_sorted) / 2

    return voxelize(testset_sorted[:splitting_point], depth), \
           voxelize(testset_sorted[splitting_point:], depth)


def calc_SAH(points, d):
    a = points[0][d]
    c = points[-1][d]
    score_best = (c - a) * len(points)
    splitting_index = 0
    for index, point in enumerate(points):
        b = point[d]
        score_cur = (b - a) * index + \
                    (c - b) * (len(points) - index)
        if score_cur < score_best:
            score_best = score_cur
            splitting_index = index
    return splitting_index


def calc_median(testset_sorted, d):
    return len(testset_sorted) / 2

def voxelize(pointset, criterion=calc_median, depth=-1, max_leaf_size=32):
    d = (depth + 1) % len(dims)
    if len(pointset) <= max_leaf_size:
        return pointset

    testset_sorted = sorted(pointset, key=lambda g: g[d])
    splitting_index = criterion(testset_sorted, d)

    return testset_sorted[splitting_index][d], voxelize(testset_sorted[:splitting_index], criterion, depth + 1), \
           voxelize(testset_sorted[splitting_index:], criterion, depth + 1)


def dist(p1, p2):
    dist = 0.0
    for d in [0, 1, 2]:
        dist = dist + (p1[d] - p2[d]) ** 2
    return dist


def find_node_by_point_tree(tree, point, depth=-1):
    if not tree:
        return None
    d = (depth + 1) % len(dims)
    # Leaf found
    if isinstance(tree[0], tuple):
        return find_node_by_point_linear(tree, point)
        #return tree[0]

    # print d, tree[0]
    search_left = point[d] < tree[0]
    best = find_node_by_point_tree(tree[1 if search_left else 2], point, depth + 1)
    if best is None:
        return None
    # There is a chance that there is a better fitting point on the other side of the hyperplane
    if (tree[0] - point[d]) ** 2 < dist(best, point):
        cand = find_node_by_point_tree(tree[1 if not search_left else 2], point, depth + 1)
        if cand is not None:
            best = cand if dist(cand, point) < dist(best, point) else best
    return best


def main():
    testpoints = [gen_random_point() for _ in xrange(0, 100)]

    for criterion in (calc_median, calc_SAH):
        print criterion.__name__
        # SAH
        start = timer()
        tree = voxelize(testset, criterion=criterion)
        end = timer()
        print "time to build", (end - start)
        # pprint(tree)

        start = timer()
        for test_point in testpoints:
            found_point_tree = find_node_by_point_tree(tree, test_point)
        end = timer()
        # print found_point_tree, dist(found_point_tree, test_point)
        print "time to find point", (end - start)

    #start = timer()
    #for test_point in testpoints:
    #    found_point_linear = find_node_by_point_linear(testset, test_point)
    #end = timer()
    #print found_point_linear, dist(found_point_linear, test_point)
    #print (end - start)


if __name__ == "__main__":
    main()
