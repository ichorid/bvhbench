import bisect
import gzip
import pickle
import random
from math import log2, floor, ceil
from pprint import pprint
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


def find_node_by_point_linear(tset, point):
    best_match = tset[0]
    best_match_dist = 0.0
    for p in tset:
        dist_cand = dist(point, p)
        if dist_cand < best_match_dist or best_match_dist == 0.0:
            best_match_dist = dist_cand
            best_match = p
    return best_match


# UNUSED
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
    # print(point[d], tree[0])
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
num_cells = 2 ** MORTON_BITS_PER_DIM


def get_quadrant(point):
    return tuple(int(floor(point[d] * num_cells)) for d in dims)


def get_morton_code(point):
    point_quadrant = get_quadrant(point)
    point_code = 0
    for d in dims:
        for k in range(0, MORTON_BITS_PER_DIM):
            assert (point_quadrant[d] >= 0)
            assert (point_quadrant[d] < 2 ** MORTON_BITS_PER_DIM)

            cp = (point_quadrant[d]>>k) & 1
            cp_shifted = cp << (len(dims) -1 -d+k*len(dims))
            point_code = point_code ^ (cp_shifted)

    return point_code


def find_outliers(points):
    outliers = []
    for d in dims:
        points_srt = sorted(points, key=lambda p: p[d])
        outliers.append((points_srt[0][d], points_srt[-1][d]))
    return outliers


def normalize_points(points):
    outliers = find_outliers(points)

    # We normalize to values slightly bigger that that of
    # the real outliers to make sure 1.0 will never appear later
    norm_coeff = tuple(1.00001 * (outliers[d][1] - outliers[d][0]) for d in dims)

    points_normalized = []
    for p in points:
        points_normalized.append(
            tuple((p[d] - outliers[d][0]) / norm_coeff[d] for d in dims))
        for d in dims:
            assert (points_normalized[-1][d] >= 0)
    return points_normalized


def find_nearest_neighbor_morton(morton_list, point):
    morton_point = get_morton_code(point)
    return bisect.bisect_left(morton_list, morton_point)


def radix_delta(mpisl, i1, i2):
    if 0 <= i1 < len(mpisl) and 0 <= i2 < len(mpisl):
        return clz(mpisl[i1][1] ^ mpisl[i2][1])
    else:
        return -1


def compute_range_upper_bound(mpisl, index, d):
    delta_min = radix_delta(mpisl, index, index - d)
    l_max = 2
    while radix_delta(mpisl, index, index + l_max * d) > delta_min:
        l_max = l_max * 2
    return l_max


def calculate_node_properties(mpisl, i):
    # Determine "d".
    # Get deltas with the node to the left and to the right of the given i.
    # If right delta is higher than the left one, return +1, otherwise -1
    # There should never be the case where the delta is the same!
    d = 1 if (radix_delta(mpisl, i, i + 1) - radix_delta(mpisl, i, i - 1)) > 0 else -1

    # Compute range end using binary search
    delta_min = radix_delta(mpisl, i, i - d)
    l_max = compute_range_upper_bound(mpisl, i, d)
    l = 0  # the length of the node domain in direction d
    for t in [int(l_max / (2 ** n)) for n in range(1, int(log2(l_max) + 1))]:
        if radix_delta(mpisl, i, i + (l + t) * d) > delta_min:
            l = l + t
    j = i + l * d  # range end

    # Find split position using binary search
    delta_node = radix_delta(mpisl, i, j)
    s = 0
    for t in [int(ceil(l / (2 ** n))) for n in range(1, int(log2(l or 1) + 1))]:
        if radix_delta(mpisl, i, i + (s + t) * d) > delta_node:
            s = s + t
    gamma = i + s * d + min(d, 0)  # split position
    #print("GAMMA CALC:", i, s, d, min(d, 0))
    split_delta = radix_delta(mpisl, gamma, gamma + 1)
    #print("BLA ", i, j, l_max * d, gamma, gamma + d, split_delta)
    assert (i <= gamma < j or j <= gamma < i)
    return j, gamma, split_delta


def morton2point(mpoint):
    point = [0, 0, 0]
    for d in dims:
        coordinate = 0.0
        for k in range(0, MORTON_BITS_PER_DIM):
            coordinate = coordinate +(((mpoint >> (len(dims) - 1 - d + k * len(dims))) & 1) << k)
        point[d] = coordinate / num_cells
    return tuple(point)


def voxelize_to_point_tuples_tree_by_morton_radix(mpisl, i, orig_points):
    j, g, split_delta = calculate_node_properties(mpisl, i)

    if min(i, j) == g:
        left_child = [orig_points[k] for k in mpisl[g][0]]
    else:
        #print("<<<<<")
        left_child = voxelize_to_point_tuples_tree_by_morton_radix(mpisl, g, orig_points)

    if max(i, j) == g + 1:
        right_child = [orig_points[k] for k in mpisl[g + 1][0]]
    else:
        #print(">>>>>")
        right_child = voxelize_to_point_tuples_tree_by_morton_radix(mpisl, g + 1, orig_points)

    assert (split_delta >= 0)  # this should never happen, as we handle leaves on upper layers
    assert (g >= 0)
    split_position = morton2point(mpisl[g][1])[len(dims) - 1 - (split_delta % len(dims))]

    return split_position, left_child, right_child


def compact_duplicates(mpisl):
    new_list = []
    for e in mpisl:
        if not new_list or new_list[-1][1] != e[1]:
            new_list.append([[e[0]], e[1]])
        else:
            new_list[-1][0].append(e[0])
    return new_list


def construct_binary_radix_tree(pl):
    mortonized_points = tuple(get_morton_code(p) for p in pl)
    mortonized_points_indexed = enumerate(mortonized_points)
    mpi_sorted = sorted(mortonized_points_indexed, key=lambda x: x[1])
    mpi_sorted_compacted = compact_duplicates(mpi_sorted)
    print(len(mpi_sorted), len(mpi_sorted_compacted))
    # for i in range(0, len(mpi_sorted_compacted) - 1):
    #    j, g = calculate_node_properties(mpi_sorted_compacted, i)
    #    # node = (leaf(i) if min(i,j) == g else node(g), leaf(g+1) if max(i,j) == g+1 else node(g+1))
    #pprint(pl)
    mrtree = voxelize_to_point_tuples_tree_by_morton_radix(mpi_sorted_compacted, 0, pl)

    # for i in range(1,100):
    #    print(radix_delta(mpi_sorted[i-1][1], mpi_sorted[i][1]))
    return mrtree


def main():
    # testpoints = [gen_random_point() for _ in xrange(0, 100)]
    testset = normalize_points(random.sample(HAND, 10))
    #testset = normalize_points(HAND)
    # testset = HAND
    testpoints = [perturbate_point(p, 0.0) for p in random.sample(testset, 10)]


    point_search_results = []

    for criterion in (calc_random, calc_SAH, calc_median):
        print(criterion.__name__)
        # SAH
        start = timer()
        tree = voxelize(testset, criterion=criterion)
        end = timer()
        print("time to build", (end - start))

        pprint(tree)

        start = timer()
        search_results = []
        for test_point in testpoints:
            search_results.append(find_node_by_point_tree(tree, test_point))
        point_search_results.append(search_results)

        end = timer()
        # print found_point_tree, dist(found_point_tree, test_point)
        print("time to find point", (end - start))
        print("\n")

    print("morton stuff")
    start = timer()
    morton_tree = construct_binary_radix_tree(testset)
    end = timer()
    print("time to build ", (end - start))

    # mortonized_points_sorted = sorted(mortonized_points)
    # m_point = mortonized_points_sorted[find_nearest_neighbor_morton(mortonized_points_sorted, testpoints[10])]
    # print(mortonized_points.index(m_point))
    start = timer()
    morton_results = []
    for test_point in testpoints:
        morton_results.append(find_node_by_point_tree(morton_tree, test_point))
    point_search_results.append(morton_results)
    end = timer()
    print("time to find point", (end - start))

    pprint(morton_tree)
    print("\n")
    for res in range(0, 10):
        print(testpoints[res])
        for c in point_search_results:
            print(c[res])
        print("\n")

    # start = timer()
    # for test_point in testpoints:
    #    found_point_linear = find_node_by_point_linear(testset, test_point)
    # end = timer()
    # print found_point_linear, dist(found_point_linear, test_point)
    # print (end - start)


if __name__ == "__main__":
    main()
