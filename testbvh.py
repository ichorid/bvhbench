import bisect
import gzip
import pickle
import random
from timeit import default_timer as timer


def clz(n):
    return 67 - len(bin(-n)) & ~n >> 64


with gzip.open("hand_small.pickle.gz", 'rb') as f:
    HAND = pickle.load(f)

set_size = 100000

dims = (0, 1, 2)

random.seed(123)


def gen_random_point(a=-1.0, b=1.0):
    return tuple(random.uniform(a, b) for _ in dims)


def perturbate_point(point, coeff=0.1):
    return tuple(point[d] * (1.0 + random.uniform(-coeff / 2, coeff / 2)) for d in dims)


# testset = tuple(gen_random_point() for _ in xrange(set_size))
testset = HAND


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


def calc_random(testset_sorted, d):
    return int(random.uniform(0, len(testset_sorted)))


def voxelize(pointset, criterion=calc_median, depth=-1, max_leaf_size=1):
    d = (depth + 1) % len(dims)
    if len(pointset) <= max_leaf_size:
        return pointset

    testset_sorted = sorted(pointset, key=lambda g: g[d])
    splitting_index = int(criterion(testset_sorted, d))

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
        # return tree[0]

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


MORTON_BITS_PER_DIM = 21


def get_quadrant(point, num_cells=2 ** MORTON_BITS_PER_DIM):
    return tuple(int(point[d] * num_cells) for d in dims)


def get_morton_code(point):
    point_quadrant = get_quadrant(point)
    for d in dims:
        point_code = 0
        for k in range(0, MORTON_BITS_PER_DIM):
            bit_d = (point_quadrant[d] >> k) & 1
            point_code = point_code ^ (bit_d << ((len(dims) - d) + k * len(dims)))
    return point_code


def find_outliers(points):
    outliers = []
    for d in dims:
        points_srt = sorted(points, key=lambda p: p[d])
        outliers.append((points_srt[0][d], points_srt[-1][d]))
    return outliers


def normalize_points(points):
    outliers = find_outliers(points)

    points_normalized = []
    for p in points:
        points_normalized.append(
            tuple((p[d] - outliers[d][0]) / (outliers[d][1] - outliers[d][0]) for d in dims))
    return points_normalized


def find_nearest_neighbor_morton(morton_list, point):
    morton_point = get_morton_code(point)
    return bisect.bisect_left(morton_list, morton_point)


def main():
    # testpoints = [gen_random_point() for _ in xrange(0, 100)]
    testset = normalize_points(HAND)
    # testset = HAND
    testpoints = [perturbate_point(p, 0.0) for p in random.sample(testset, 10000)]
    mortonized_points = tuple(get_morton_code(p) for p in testpoints)
    mortonized_points_sorted = sorted(mortonized_points)
    m_point = mortonized_points_sorted[find_nearest_neighbor_morton(mortonized_points_sorted, testpoints[10])]
    print(mortonized_points.index(m_point))

    for criterion in (calc_median, calc_SAH, calc_random):
        print(criterion.__name__)
        # SAH
        start = timer()
        tree = voxelize(testset, criterion=criterion)
        end = timer()
        print("time to build", (end - start))
        # pprint(tree)

        start = timer()
        for test_point in testpoints:
            found_point_tree = find_node_by_point_tree(tree, test_point)
        end = timer()
        # print found_point_tree, dist(found_point_tree, test_point)
        print("time to find point", (end - start))

    # start = timer()
    # for test_point in testpoints:
    #    found_point_linear = find_node_by_point_linear(testset, test_point)
    # end = timer()
    # print found_point_linear, dist(found_point_linear, test_point)
    # print (end - start)


if __name__ == "__main__":
    main()
