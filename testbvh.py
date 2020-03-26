import bisect
import gzip
import pickle
import random
from copy import deepcopy
from itertools import permutations
from math import log2, floor, ceil
from pprint import pprint
from timeit import default_timer as timer

import attr
from numpy.ma import empty

set_size = 100000

dims = (0, 1, 2)

random.seed(123)

MORTON_BITS_PER_DIM = 10
num_cells = 2 ** MORTON_BITS_PER_DIM
HIGHEST_MORTON_POINT = 2 ** (MORTON_BITS_PER_DIM * len(dims)) - 1

RULER = "012" * MORTON_BITS_PER_DIM

# FIXME: this implementation can't handle leafs with points that have exactly the same coordinate.

dim_mask = (int("001" * MORTON_BITS_PER_DIM, 2),
            int("010" * MORTON_BITS_PER_DIM, 2),
            int("100" * MORTON_BITS_PER_DIM, 2))

DEBUG = False

TREELET_SIZE = 7
K_BIT_PERMUTATIONS = list(range(0, TREELET_SIZE + 1))
K_BIT_PERMUTATIONS[0] = []
for num_ones in range(1, TREELET_SIZE + 1):
    K_BIT_PERMUTATIONS[num_ones] = [int("".join(str(nums) for nums in tp), 2) for tp in
                                    {k for k in permutations("1" * num_ones + "0" * (TREELET_SIZE - num_ones))}]
    print(K_BIT_PERMUTATIONS[num_ones])

zz = set()
for z in K_BIT_PERMUTATIONS:
    for zzz in z:
        zz.add(zzz)


print (len(zz))


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


@attr.s
class Node:
    d = attr.ib(type=int)
    split = attr.ib(type=float)
    lc = attr.ib()
    rc = attr.ib()

    def print_subtree(self, depth):
        print(" " * depth, self.d, self.split)
        for n in self.lc, self.rc:
            if n and isinstance(n, self.__class__):
                n.print_subtree(depth + 1)
            else:
                print(" " * (depth + 1), n)


def clz(n):
    return 67 - len(bin(-n)) & ~n >> 64


with gzip.open("hand_small.pickle.gz", 'rb') as f:
    HAND = pickle.load(f)


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
    return len(testset_sorted) // 2


def calc_random(testset_sorted, d):
    return int(random.uniform(0, len(testset_sorted)))


def voxelize(pointset, criterion=calc_median, depth=-1, max_leaf_size=1):
    d = (depth + 1) % len(dims)
    if len(pointset) <= max_leaf_size:
        return pointset

    testset_sorted = sorted(pointset, key=lambda g: g[d])
    splitting_index = int(criterion(testset_sorted, d))
    # Dumb hack to fix duplicate coordinates problem.
    while splitting_index > 0 and testset_sorted[splitting_index][d] == testset_sorted[splitting_index - 1][d]:
        splitting_index = splitting_index - 1

    return Node(
        d=d,
        split=testset_sorted[splitting_index][d],
        lc=voxelize(testset_sorted[:splitting_index], criterion, depth + 1),
        rc=voxelize(testset_sorted[splitting_index:], criterion, depth + 1))


def dist(p1, p2):
    dist = 0.0
    for d in [0, 1, 2]:
        dist = dist + (p1[d] - p2[d]) ** 2
    return dist


def find_node_by_point_tree(node, point, depth=-1):
    if not node:
        return None
    # Leaf found
    if isinstance(node, list):
        return find_node_by_point_linear(node, point)
        # return tree[0]

    d = node.d
    # print d, tree[0]
    # print(point[d], tree[0])
    search_left = point[d] < node.split
    best = find_node_by_point_tree(node.lc if search_left else node.rc, point, depth + 1)
    if best is None:
        return None
    # There is a chance that there is a better fitting point on the other side of the hyperplane
    if (node.split - point[d]) ** 2 < dist(best, point):
        cand = find_node_by_point_tree(node.lc if search_left else node.rc, point, depth + 1)
        if cand is not None:
            best = cand if dist(cand, point) < dist(best, point) else best
    return best


def find_node_by_point_tree_flat(nodes_list, point, index, depth=-1):
    node = nodes_list[index]

    d = node.d
    # print d, tree[0]
    # print(point[d], tree[0])
    search_left = point[d] < node.split
    next_search_node_ind = node.lc if search_left else node.rc

    # Leaf found
    if isinstance(next_search_node_ind, list):
        return find_node_by_point_linear(next_search_node_ind, point)
        # return tree[0]

    best = find_node_by_point_tree_flat(nodes_list, point, next_search_node_ind, depth + 1)
    if best is None:
        return None
    # There is a chance that there is a better fitting point on the other side of the hyperplane
    if (node.split - point[d]) ** 2 < dist(best, point):
        cand = find_node_by_point_tree_flat(nodes_list, point, next_search_node_ind, depth + 1)
        if cand is not None:
            best = cand if dist(cand, point) < dist(best, point) else best
    return best


def get_quadrant(point):
    return tuple(int(floor(point[d] * num_cells)) for d in dims)


def get_morton_code(point):
    point_quadrant = get_quadrant(point)
    point_code = 0
    for d in dims:
        assert (point_quadrant[d] >= 0)
        assert (point_quadrant[d] < 2 ** MORTON_BITS_PER_DIM)
        for k in range(0, MORTON_BITS_PER_DIM):
            cp = (point_quadrant[d] >> k) & 1
            cp_shifted = cp << (len(dims) - 1 - d + k * len(dims))
            point_code = point_code ^ cp_shifted

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


def radix_delta(mpisl, i, j):
    if 0 <= j < len(mpisl):
        return clz(mpisl[i][1] ^ mpisl[j][1]) - 1  # CAUTION! 64 is not 63 !!!
    else:
        return -1


def compute_range_upper_bound(mpisl, index, d):
    delta_min = radix_delta(mpisl, index, index - d)
    l_max = 2
    while radix_delta(mpisl, index, index + l_max * d) > delta_min:
        l_max = l_max * 2
    return l_max


def morton2point(mpoint):
    point = [0, 0, 0]
    for d in dims:
        coordinate = 0.0
        for k in range(0, MORTON_BITS_PER_DIM):
            coordinate = coordinate + (((mpoint >> (len(dims) - 1 - d + k * len(dims))) & 1) << k)
        point[d] = coordinate / num_cells
    return tuple(point)


def calculate_node_properties(mpisl, i):
    debug_print("\n<<>>")
    debug_print("node index ", i)
    debug_print("left: ", "{0:64b}".format(mpisl[i - 1][1]), radix_delta(mpisl, i, i - 1))
    debug_print("node: ", "{0:64b}".format(mpisl[i][1]))
    debug_print("right:", "{0:64b}".format(mpisl[i + 1][1]), radix_delta(mpisl, i, i + 1))
    # Determine "d".
    # Get deltas with the node to the left and to the right of the given i.
    # If right delta is higher than the left one, return +1, otherwise -1
    # There should never be the case where the delta is the same!
    d = 1 if (radix_delta(mpisl, i, i + 1) - radix_delta(mpisl, i, i - 1)) > 0 else -1

    # Compute range end using binary search
    delta_min = radix_delta(mpisl, i, i - d)
    l_max = compute_range_upper_bound(mpisl, i, d)
    if DEBUG:
        print("l_max", l_max)
        if d > 0:
            for ind in range(i, min(i + l_max + 1, len(mpisl))):
                print("node:", ind, ":{0:64b}".format(mpisl[ind][1]))
        else:
            for ind in range(max(0, i - l_max), i + 1):
                print("node:", ind, ":{0:64b}".format(mpisl[ind][1]))

    l = 0  # the length of the node domain in direction d
    for t in [int(l_max / (2 ** n)) for n in range(1, int(log2(l_max) + 1))]:
        if radix_delta(mpisl, i, i + (l + t) * d) > delta_min:
            l = l + t
    j = i + l * d  # range end

    # Find split position using binary search
    delta_node = radix_delta(mpisl, i, j)
    s = 0
    # rng = [int(ceil(l / (2 ** n))) for n in range(1, int(log2(l or 1) + 1))]

    t_real = l / 2
    while True:
        t = ceil(t_real)
        debug_print("binary_search_step", i + (s + t) * d)
        if radix_delta(mpisl, i, i + (s + t) * d) > delta_node:
            s = s + t
        if t == 1:
            break
        t_real = t_real / 2

    gamma = i + s * d + min(d, 0)  # split position
    debug_print("s g", s, gamma)
    # print("GAMMA CALC:", i, s, d, min(d, 0))
    split_delta = radix_delta(mpisl, gamma, gamma + 1)
    assert (split_delta == radix_delta(mpisl, i, j))
    # print("BLA ", i, j, l_max * d, gamma, gamma + d, split_delta)
    assert (i <= gamma < j or j <= gamma < i)
    return j, gamma, split_delta


def gen_flat_tree_morton(mpisl, orig_points):
    result = []
    check = set()
    for i in range(0, len(mpisl) - 1):
        j, g, split_delta = calculate_node_properties(mpisl, i)

        if min(i, j) == g:
            left_child = [orig_points[k] for k in mpisl[g][0]]
            for k in left_child:
                check.add(k)
        else:
            left_child = g

        if max(i, j) == g + 1:
            right_child = [orig_points[k] for k in mpisl[g + 1][0]]
            for k in right_child:
                check.add(k)
        else:
            right_child = g + 1

        split_dim = split_delta % len(dims)

        mask_length = split_delta + 1
        split_surface_prefix_mask = ((1 << mask_length) - 1) << (63 - mask_length)
        split_surface_morton_code = morton2point(mpisl[g + 1][1] & split_surface_prefix_mask)
        split_position = split_surface_morton_code[split_dim]

        result.append(Node(d=split_dim, split=split_position, lc=left_child, rc=right_child))
        debug_print(i, result[-1], g, g + 1)
    print("SET SIZE", len(check))
    return result


def voxelize_to_point_tuples_tree_by_morton_radix(mpisl, i, orig_points):
    j, g, split_delta = calculate_node_properties(mpisl, i)
    debug_print("POINTS around split ")
    debug_print("{0:64b}".format(mpisl[g - 1][1]), morton2point(mpisl[g - 1][1]))
    debug_print("{0:64b}".format(mpisl[g][1]), morton2point(mpisl[g][1]))
    debug_print("{0:64b}".format(mpisl[g + 1][1]), morton2point(mpisl[g + 1][1]))

    if min(i, j) == g:
        left_child = [orig_points[k] for k in mpisl[g][0]]
    else:
        # left_child = None
        left_child = voxelize_to_point_tuples_tree_by_morton_radix(mpisl, g, orig_points)
        # print("<<<<<")

    if max(i, j) == g + 1:
        right_child = [orig_points[k] for k in mpisl[g + 1][0]]
    else:
        # right_child = None
        right_child = voxelize_to_point_tuples_tree_by_morton_radix(mpisl, g + 1, orig_points)
        # print(">>>>>")

    assert (split_delta >= 0)  # this should never happen, as we handle leaves on upper layers
    assert (g >= 0)
    split_dim = (split_delta) % len(dims)

    # split_position = orig_points[mpisl[g + 1][0][0]][split_dim]  # !!!!!!! +1 !!!!!

    d = 1 if j > i else -1
    mask_length = split_delta + 1
    split_surface_prefix_mask = ((1 << mask_length) - 1) << (63 - mask_length)
    split_surface_morton_code = morton2point(mpisl[g + 1][1] & split_surface_prefix_mask)
    split_position = split_surface_morton_code[split_dim]

    if DEBUG:
        # Check that the points in the i,j range really belong to spatial constraints
        if j > i:
            left_range, right_range = [z for z in range(i, g + 1)], [z for z in range(g + 1, j + 1)]
        else:
            left_range, right_range = [z for z in range(j, g + 1)], [z for z in range(g + 1, i + 1)]

        # left_subnodes = [orig_points[mpisl[ind][0][0]] for ind in left_range]
        # right_subnodes = [orig_points[mpisl[ind][0][0]] for ind in right_range]
        left_subnodes = [morton2point(mpisl[ind][1]) for ind in left_range]
        right_subnodes = [morton2point(mpisl[ind][1]) for ind in right_range]
        print("\n")
        print("INDEX: ", i, j, g, split_delta, d)
        print("node l: ", len(left_range), left_range)
        print("node r: ", len(right_range), right_range)

        print("Left subnodes ")
        for z in left_subnodes:
            print(z)
            print("{0:63b}".format(get_morton_code(z)))
            print(RULER)
        print("Split:", split_dim, split_position)
        print("Right subnodes")
        for z in right_subnodes:
            print("{0:63b}".format(get_morton_code(z)))
            print(RULER)
            print(z)

        for z in left_subnodes:
            if not z[split_dim] < split_position:
                print(" ERROR ")
                exit(1)
        for z in right_subnodes:
            # assert(z[split_dim] >= split_position)
            if not z[split_dim] >= split_position:
                print(" ERROR ")
                exit(1)

    return Node(d=split_dim, split=split_position, lc=left_child, rc=right_child)


def compact_duplicates(mpisl):
    new_list = []
    for e in mpisl:
        if not new_list or new_list[-1][1] != e[1]:
            new_list.append([[e[0]], e[1]])
        else:
            new_list[-1][0].append(e[0])
    for i, _ in enumerate(new_list):
        if i > 0:
            assert (new_list[i][1] > new_list[i - 1][1])
    return new_list


def convert_points_to_morton_codes(pl):
    mortonized_points = tuple(get_morton_code(p) for p in pl)
    mortonized_points_indexed = enumerate(mortonized_points)
    mpi_sorted = sorted(mortonized_points_indexed, key=lambda x: x[1])
    mpi_sorted_compacted = compact_duplicates(mpi_sorted)
    return mpi_sorted_compacted


def construct_binary_tree_by_morton_codes(pl):
    morton_points = convert_points_to_morton_codes(pl)
    mrtree = voxelize_to_point_tuples_tree_by_morton_radix(morton_points, 0, pl)
    # for i in range(0, len(mpi_sorted_compacted) - 1):
    #    j, g = calculate_node_properties(mpi_sorted_compacted, i)
    #    # node = (leaf(i) if min(i,j) == g else node(g), leaf(g+1) if max(i,j) == g+1 else node(g+1))
    # pprint(pl)
    check_binary_tree(mrtree)

    # for i in range(1,100):
    #    print(radix_delta(mpi_sorted[i-1][1], mpi_sorted[i][1]))
    return mrtree


def construct_flat_tree(pl):
    morton_points = convert_points_to_morton_codes(pl)
    flat = gen_flat_tree_morton(morton_points, pl)
    check_flat_tree(flat)
    return flat


def check_constraints(point, constraints):
    for d, (coord_l, coord_r) in enumerate(constraints):
        p = point[d]
        left_ok, right_ok = True, True
        if coord_l is not None:
            left_ok = p >= coord_l
        if coord_r is not None:
            right_ok = p < coord_r
        if not left_ok or not right_ok:
            print("ERROR", point, constraints, (d, coord_l, coord_r))
            return True
    return False


def check_flat_tree(nodes_list: list):
    for node in nodes_list:
        constraints = [[None, None], [None, None], [None, None]]
        for child, left in ((node.lc, 1), (node.rc, 0)):  # left node constrained from right, and vice-versa
            old_dim_constraint = constraints[node.d][left]
            if old_dim_constraint is not None:
                if left:
                    assert (node.split < old_dim_constraint)
                else:
                    assert (node.split >= old_dim_constraint)

            new_constraints = deepcopy(constraints)
            new_constraints[node.d][left] = node.split
            if isinstance(child, list):
                for point in child:
                    if check_constraints(point, new_constraints):
                        print(node)
                        exit(1)


def check_binary_tree(node, constraints: list = None):
    if constraints is None:
        constraints = [[None, None], [None, None], [None, None]]
    if isinstance(node, list):
        for point in node:
            check_constraints(point, constraints)
        return

    for child, left in ((node.lc, 1), (node.rc, 0)):  # left node constrained from the right, and vice-versa
        old_dim_constraint = constraints[node.d][left]
        if old_dim_constraint is not None:
            if left:
                assert (node.split < old_dim_constraint)
            else:
                assert (node.split >= old_dim_constraint)

        new_constraints = deepcopy(constraints)
        new_constraints[node.d][left] = node.split
        check_binary_tree(child, new_constraints)


def surface_area_morton(point_a, point_b):
    box_dims = tuple(abs((point_a & dim_mask[d]) - (point_b & dim_mask[d])) for d in dims)
    surface_area = box_dims[0] * box_dims[1] + box_dims[0] * box_dims[2] + box_dims[1] * box_dims[2]
    return surface_area


def get_bits_from_bitmask(bm):
    return [int(x) for x in '{:032b}'.format(bm)]


def area_union_of_AABBs(L, s):
    # We use Morton codes to calculate the surface area.
    lowest, highest = HIGHEST_MORTON_POINT, 0
    for i in get_bits_from_bitmask(s):
        point = L[i]
        if point < lowest:
            lowest = point
        if point > highest:
            highest = point
    area = surface_area_morton(lowest, highest)
    return area


def SAH_Cost(leaf):
    raise NotImplementedError()


def optimize_treelet(treelet):
    L = treelet
    n = len(L)
    # Calculate surface area for each subset
    a = empty(2 ** n, type=float)
    for s in range(1, 2 ** n):
        a[s] = area_union_of_AABBs(L, s)

    # Initialize costs of individual leaves
    c_opt = empty(2 ** n, type=float)
    for i in range(0, n):
        c_opt[2 ** i] = SAH_Cost(L[i])

    # Optimize every subset of leaves
    for k in range(2, n + 1):
        for s in range(1, 2 ** n):
            pass


def main():
    test_mpisl = [([0.0], 1 << 62), ([0.0], 1 << 0)]
    print(radix_delta(test_mpisl, 0, 1))
    print("{0:64b}".format(test_mpisl[1][1]))
    print("{0:64b}".format(test_mpisl[0][1]))

    print("{0:64b}".format(test_mpisl[0][1]))
    point = (0.99, 0.8, 0.5)
    point_m = get_morton_code(point)
    print(point_m, get_morton_code(morton2point(get_morton_code(morton2point(point_m)))))
    # pprint(get_morton_code((0,0,0)))
    # print("{0:64b}".format(get_morton_code((00.0000006*6.1,00.0000006*6.7,0.0000006))))
    # testpoints = [gen_random_point() for _ in xrange(0, 100)]
    # WARNING: we perturbate the points to guarantee that there will be no duplicates
    testset = normalize_points(random.sample(HAND, 100))
    # testset = normalize_points(HAND)
    # testset = normalize_points([perturbate_point(p, 0.000001) for p in random.sample(HAND, 10000)])
    # testset = normalize_points(HAND)
    # testset = HAND
    testpoints = [perturbate_point(p, 0.0) for p in random.sample(testset, 100)]

    point_search_results = []

    for criterion in (calc_random, calc_SAH, calc_median):
        break
        print(criterion.__name__)
        # SAH
        start = timer()
        tree = voxelize(testset, criterion=criterion)
        end = timer()
        print("time to build", (end - start))
        # tree.print_subtree(0)

        check_binary_tree(tree)
        start = timer()
        search_results = []
        for test_point in testpoints:
            search_results.append(find_node_by_point_tree(tree, test_point))
        point_search_results.append(search_results)

        end = timer()
        # print found_point_tree, dist(found_point_tree, test_point)
        print("time to find point", (end - start))
        print("\n")

    print("construct morton-based binary tree")
    start = timer()
    morton_tree = construct_binary_tree_by_morton_codes(testset)
    end = timer()
    print("time to build ", (end - start))

    print("construct morton-based flat tree")
    start = timer()
    flat_tree = construct_flat_tree(testset)
    end = timer()
    print("time to build ", (end - start))

    return

    # mortonized_points_sorted = sorted(mortonized_points)
    # m_point = mortonized_points_sorted[find_nearest_neighbor_morton(mortonized_points_sorted, testpoints[10])]
    # print(mortonized_points.index(m_point))
    start = timer()
    morton_results = []
    for test_point in testpoints[:100]:
        morton_results.append(find_node_by_point_tree_flat(flat_tree, test_point, 0))
        # morton_results.append(find_node_by_point_tree(morton_tree, test_point))
    point_search_results.append(morton_results)
    end = timer()
    print("time to find point", (end - start))

    # morton_tree.print_subtree(0)
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
