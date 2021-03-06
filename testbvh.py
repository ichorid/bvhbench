import bisect
import gzip
import pickle
import random
from copy import deepcopy, copy
from heapq import heappop, heappush
from itertools import permutations
from math import log2, floor, ceil, inf
from pprint import pprint
from timeit import default_timer as timer

import attr
from numpy import empty

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

TREELET_SIZE = 8
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

print(len(zz))

TERMINAL_MASKS = tuple(1 << n for n in range(0, TREELET_SIZE))


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


SEARCH_INVOCATIONS = 0

@attr.s
class Node:
    index = attr.ib(type=int)
    d = attr.ib(type=int)
    split = attr.ib(type=float)
    lc = attr.ib()
    rc = attr.ib()
    num_points = attr.ib(type=int, default=0)
    aabb_surface = attr.ib(type=float, default=0.0)
    sah_cost = attr.ib(type=float, default=0.0)
    aabb = attr.ib(type=tuple, default=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))

    def print_subtree(self, depth):
        print(" " * depth, self.d, self.split)
        for n in self.lc, self.rc:
            if n and isinstance(n, self.__class__):
                n.print_subtree(depth + 1)
            else:
                print(" " * (depth + 1), n)


@attr.s
class Leaf:
    num_points = attr.ib(type=int, default=0)
    aabb_surface = attr.ib(type=float, default=0.0)
    sah_cost = attr.ib(type=float, default=0.0)
    aabb = attr.ib(type=tuple, default=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    points = attr.ib(type=list, default=list())


def make_leaf_from_points(points: list):
    aabb = AABB_from_points(points)
    aabb_surface = 0.0000001 + surface_area(*aabb)
    return Leaf(
        num_points=len(points),
        aabb=aabb,
        aabb_surface=aabb_surface,
        sah_cost=len(points) * aabb_surface,
        points=points
    )


def clz(n):
    return 67 - len(bin(-n)) & ~n >> 64


def mask_to_index(mask):
    return (64 - clz(mask)) - 1


with gzip.open("sibenik_small.pickle.gzip", 'rb') as f:
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
    global SEARCH_INVOCATIONS
    SEARCH_INVOCATIONS = SEARCH_INVOCATIONS + 1
    node = nodes_list[index]

    d = node.d
    # print d, tree[0]
    # print(point[d], tree[0])
    search_left = point[d] < node.split
    next_search_node_ind = node.lc if search_left else node.rc

    # Leaf found
    if isinstance(next_search_node_ind, Leaf):
        return find_node_by_point_linear(next_search_node_ind.points, point)
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
    tree = []
    for i in range(0, len(mpisl) - 1):
        j, g, split_delta = calculate_node_properties(mpisl, i)

        # Left child
        if min(i, j) == g:
            left_child = make_leaf_from_points([orig_points[k] for k in mpisl[g][0]])
        else:
            left_child = g

        # Right child
        if max(i, j) == g + 1:
            right_child = make_leaf_from_points([orig_points[k] for k in mpisl[g + 1][0]])
        else:
            right_child = g + 1

        # Splitting plane position
        split_dim = split_delta % len(dims)

        mask_length = split_delta + 1
        split_surface_prefix_mask = ((1 << mask_length) - 1) << (63 - mask_length)
        split_surface_morton_code = morton2point(mpisl[g + 1][1] & split_surface_prefix_mask)
        split_position = split_surface_morton_code[split_dim]

        #left_morton_code = mpisl[min(i, j)][1]
        #right_morton_code = mpisl[min(i, j)][1]
        #left_border = morton2point(left_morton_code&split_surface_prefix_mask)
        #right_border = morton2point((right_morton_code&split_surface_prefix_mask) | ((right_morton_code&split_surface_prefix_mask) -1))
        #bounding_box = AABB_from_points([left_border, right_border])
        #print (left_border,right_border)


        indices = [mpisl[k][0] for k in range(min(i,j), max(i,j)+1)]
        bounding_box = AABB_from_points([orig_points[p] for sublist in indices for p in sublist])

        # bounding_box = (left_border, right_border) if j > i else (right_border, left_border)

        node_surface_area = surface_area(*bounding_box)
        # split_length = abs(morton2point(mpisl[i][1])[split_dim] - morton2point(mpisl[j][1])[split_dim])

        # if isinstance(left_child, list):
        # leaf_SAH_costs[g] = len(left_child) * abs(left_child[0][split_dim] - split_position) / split_length

        # if isinstance(right_child, list):
        # leaf_SAH_costs[g + 1] = len(right_child) * abs(right_child[0][split_dim] - split_position) / split_length

        lc_num_points = abs(min(i, j) - g) + 1
        rc_num_points = abs(max(i, j) - g) + 1
        split_length = bounding_box[0][split_dim] - bounding_box[1][split_dim]
        lc_part = split_position - bounding_box[0][split_dim]
        rc_part = split_length - lc_part
        SAH_left = node_surface_area * lc_num_points * lc_part / split_length
        SAH_right = node_surface_area * rc_num_points * rc_part / split_length

        for d in dims:
            assert (bounding_box[0][d] <= bounding_box[1][d])

        tree.append(
            Node(
                index=i,
                d=split_dim,
                split=split_position,
                lc=left_child,
                rc=right_child,
                aabb=bounding_box,
                num_points=abs(i - j) + 1,
                aabb_surface=node_surface_area,
                sah_cost=SAH_left + SAH_right
            ))
        debug_print(i, tree[-1], g, g + 1)
    return tree


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


def get_treelet_by_top_parent_index(flat_tree, index):
    treelet_leaves = []
    treelet_nodes = [(1.0 / flat_tree[index].sah_cost, flat_tree[index])]

    internal_nodes_indexes = []

    while (len(treelet_leaves) + len(treelet_nodes)) < TREELET_SIZE and treelet_nodes:
        node = heappop(treelet_nodes)[1]
        internal_nodes_indexes.append(node.index)
        left_is_leaf, right_is_leaf = isinstance(node.lc, Leaf), isinstance(node.rc, Leaf)
        lc = node.lc if left_is_leaf else flat_tree[node.lc]
        rc = node.rc if right_is_leaf else flat_tree[node.rc]
        if left_is_leaf and right_is_leaf:
            treelet_leaves.extend([lc, rc])
        else:
            if left_is_leaf:
                treelet_leaves.append(lc)
                heappush(treelet_nodes, (1.0 / rc.sah_cost, rc))
            elif right_is_leaf:
                treelet_leaves.append(rc)
                heappush(treelet_nodes, (1.0 / lc.sah_cost, lc))
            else:
                debug_print(lc.sah_cost)
                if (not lc.sah_cost >0.0):
                    print ("PSAH", repr(lc.sah_cost))
                    print (lc)
                assert (lc.sah_cost > 0.0)
                heappush(treelet_nodes, (1.0 / lc.sah_cost, lc))
                heappush(treelet_nodes, (1.0 / rc.sah_cost, rc))

    result_list = [*treelet_leaves, *[n[1] for n in treelet_nodes]]
    assert (len(set(id(t) for t in result_list)) == len(result_list))
    for i in internal_nodes_indexes:
        assert (i not in [n[1].index for n in treelet_nodes])
    return result_list, internal_nodes_indexes


def update_SAH_costs(flat_tree, index=0):
    node = flat_tree[index]
    sah_node = 1.2 * node.aabb_surface
    for child in node.lc, node.rc:
        sah_child = child.aabb_surface * child.num_points if isinstance(child, Leaf) else update_SAH_costs(flat_tree,
                                                                                                           index=child)
        sah_node += sah_child
    node.sah_cost = sah_node
    return sah_node

def optimize_with_treelets(flat_tree):

    inds = list(range(0, len(flat_tree)))
    random.shuffle(inds)
    for i in inds:
        treelet_leaves, internal_nodes_indexes = get_treelet_by_top_parent_index(flat_tree, index=i)
        if len(treelet_leaves) < TREELET_SIZE:
            continue
        #print(i)

        # for n in internal_nodes_indexes:
        #    print(flat_tree[n])
        c_opt, p_opt = optimize_treelet(treelet_leaves)
        assert (internal_nodes_indexes[0] == i)
        backtrack_optimized_treelet(flat_tree, internal_nodes_indexes, treelet_leaves, p_opt)
        # for n in internal_nodes_indexes:
        #    print(flat_tree[n])
        #check_flat_tree(flat_tree)

    update_SAH_costs(flat_tree)

def construct_flat_tree(pl):
    morton_points = convert_points_to_morton_codes(pl)
    flat_tree = gen_flat_tree_morton(morton_points, pl)
    update_SAH_costs(flat_tree)
    #check_flat_tree(flat_tree)
    original_sah_cost = flat_tree[0].sah_cost

    if DEBUG:
        for n in flat_tree:
            lc = n.lc if isinstance(n.lc, Leaf) else flat_tree[n.lc]
            rc = n.rc if isinstance(n.rc, Leaf) else flat_tree[n.rc]
            internal_node_from_nodes(lc, rc)

    optimize_with_treelets(flat_tree)

    print ("ORIG SAH: ", original_sah_cost)
    print ("UPDATED SAH: ", flat_tree[0].sah_cost)
    return flat_tree


def check_constraints(point, constraints):
    coord_l, coord_r = constraints
    for d in dims:
        p = point[d]
        left_ok, right_ok = True, True
        if coord_l[d] is not None:
            left_ok = p >= coord_l[d]
        if coord_r[d] is not None:
            right_ok = p < coord_r[d]
        if not left_ok or not right_ok:
            print("ERROR", point, constraints, (d, coord_l, coord_r))
            return True
    return False

def check_aabbs(parent_aabb, child_aabb):
    return AABB_from_points([*parent_aabb, *child_aabb]) != parent_aabb

def check_node_constraints(node):
    constraints = [[None, None, None], [None, None, None]]
    for child, left in ((node.lc, 1), (node.rc, 0)):  # left node constrained from right, and vice-versa
        old_dim_constraint = constraints[left][node.d]
        if old_dim_constraint is not None:
            if left:
                assert (node.split < old_dim_constraint)
            else:
                assert (node.split >= old_dim_constraint)

        new_constraints = deepcopy(constraints)
        new_constraints[left][node.d] = node.split
        if isinstance(child, Leaf):
            for point in child.points:
                if check_constraints(point, new_constraints):
                    print("BAD NODE: ", node)
                    exit(1)


def check_flat_tree(nodes_list: list):
    for node in nodes_list:
        if node is None:
            continue
        lc = node.lc if isinstance(node.lc, Leaf) else nodes_list[node.lc]
        rc = node.rc if isinstance(node.rc, Leaf) else nodes_list[node.rc]
        internal_node_from_nodes(lc, rc)
        if check_aabbs(node.aabb, AABB_from_points([*lc.aabb, *rc.aabb])):
            print ("NODE: ", node.index, node.aabb)
            print ("NODE: ", node.index, AABB_from_points([*lc.aabb, *rc.aabb]))

        check_node_constraints(node)



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


def surface_area(point_a, point_b):
    box_dims = [0.0, 0.0, 0.0]
    for d in dims:
        box_dims[d] = abs(point_a[d] - point_b[d])
    return box_dims[0] * box_dims[1] + box_dims[0] * box_dims[2] + box_dims[1] * box_dims[2]


def get_bits_from_bitmask(bm):
    result = []
    i = 0
    while bm != 0:
        if 1 == bm & 1:
            result.append(i)
        i += 1
        bm = bm >> 1
    return result


def AABB_from_points(points_list):
    corner_a, corner_b = list(points_list[0]), list(points_list[0])
    for point in points_list:
        for d in dims:
            coord = point[d]
            if coord < corner_a[d]:
                corner_a[d] = coord
            if coord > corner_b[d]:
                corner_b[d] = coord
    for d in dims:
        assert (corner_a[d] <= corner_b[d])
    return tuple(corner_a), tuple(corner_b)


def aabb_intersection(a, b):
    for d in dims:
        if a[1][d] < b[0][d]:
            return False
        if b[1][d] < a[0][d]:
            return False
    return True


def optimize_treelet(treelet):
    L = treelet
    n = len(L)
    assert (n == TREELET_SIZE)
    p_opt = {}
    # Calculate surface area for each subset
    a = empty(2 ** n, dtype=float)
    aabb_union = {}
    for s in range(1, 2 ** n):
        # Get area union of AABBs
        points_list = []
        for i in get_bits_from_bitmask(s):
            node = L[i]
            points_list.extend(node.aabb)
        aabb = AABB_from_points(points_list)
        a[s] = surface_area(*aabb)
        aabb_union[s] = aabb

    # Initialize costs of individual leaves
    c_opt = empty(2 ** n, dtype=float)
    for i in range(0, n):
        c_opt[2 ** i] = L[i].sah_cost
        # print(L[i].sah_cost)
        # print("LEAF: {0:05b}".format(2**i),  c_opt[2**i])

    # Optimize every subset of leaves
    for k in range(2, n + 1):
        for s in K_BIT_PERMUTATIONS[k]:
            # Try each way of partitioning the leaves
            c_s, p_s = inf, 0
            delta = (s - 1) & s
            p = (-delta) & s
            assert (p != 0)
            while p != 0:
                c = c_opt[p] + c_opt[s ^ p]
                if aabb_intersection(aabb_union[p], aabb_union[s ^ p]):
                    c = inf
                # print(p, "{0:8b}".format(p), " {0:8b}".format(s^p))
                if c < c_s:
                    c_s, p_s = c, p
                # print ("{0:05b}".format(s), "{0:05b}".format(p), "{0:05b}".format(s^p), c)
                p = (p - delta) & s
            # Calculate final SAH cost
            # TODO: implement collapsing the treelet into a single leaf
            c_opt[s] = 1.2 * a[s] + c_s
            if p_opt.get(s) is not None:
                raise
            p_opt[s] = p_s
            # print("OPT:", "{0:05b}".format(s), "{0:05b}".format(p_s), c_opt[s])

    assert (c_opt[2 ** n - 1] != inf)
    return c_opt, p_opt


def get_split_by_aabbs(a, b):
    # Returns dimension, position and direction based on the _leftmost_ coordinate of the _rightmost_ box
    for d in dims:
        # Straight check
        left_border = a[1][d]
        right_border = b[0][d]
        if left_border < right_border:
            return d, right_border, 1

        # Reversed check
        left_border = b[1][d]
        right_border = a[0][d]
        if left_border < right_border:
            return d, right_border, -1

    raise Exception()


# print (get_split_by_aabbs(((0.4,0.4,0.4),(0.4,0.4,0.4)),((0.3,0.3,0.3),(0.3,0.3,0.3))))
# exit(1)

def internal_node_from_nodes(leaf_a, leaf_b):
    aabb = AABB_from_points([*leaf_a.aabb, *leaf_b.aabb])
    aabb_surface = surface_area(*aabb)
    num_points = leaf_a.num_points + leaf_b.num_points

    split_dim, split_position, split_dir = get_split_by_aabbs(leaf_a.aabb, leaf_b.aabb)
    # Is it really necessary?
    left_leaf, right_leaf = (leaf_a, leaf_b) if split_dir > 0 else (leaf_b, leaf_a)
    node  = Node(lc=left_leaf if isinstance(left_leaf, Leaf) else left_leaf.index,
                rc=right_leaf if isinstance(right_leaf, Leaf) else right_leaf.index,
                num_points=num_points,
                aabb=aabb,
                aabb_surface=aabb_surface,
                sah_cost=aabb_surface,
                d=split_dim,
                split=split_position,
                index=-1)
    check_node_constraints(node)
    return node


def backtrack_optimized_treelet(flat_tree, internal_nodes_indexes, treelet_leaves, p_opt):
    internal_nodes_indexes = copy(internal_nodes_indexes)
    #for x in internal_nodes_indexes:
    #    print(" ", flat_tree[x])
    internal_nodes_to_process = [(2 ** len(treelet_leaves) - 1, internal_nodes_indexes.pop(0))]  # Starting from the top
    known_leaves_mask = 0
    processed_nodes_indexes = {}

    while internal_nodes_to_process:
        s, node_ind = internal_nodes_to_process[-1]
        p = p_opt[s]
        part_mask, anti_mask = p, s ^ p

        if s == (s & known_leaves_mask):
            # all leaves for this internal node are known
            leaf_a = treelet_leaves[mask_to_index(part_mask)] if part_mask in TERMINAL_MASKS else flat_tree[
                processed_nodes_indexes[part_mask]]
            leaf_b = treelet_leaves[mask_to_index(anti_mask)] if anti_mask in TERMINAL_MASKS else flat_tree[
                processed_nodes_indexes[anti_mask]]
            assert (isinstance(leaf_a, Leaf) or isinstance(leaf_a, Node))
            assert (isinstance(leaf_b, Leaf) or isinstance(leaf_b, Node))
            flat_tree[node_ind] = internal_node_from_nodes(leaf_a, leaf_b)
            flat_tree[node_ind].index = node_ind
            processed_nodes_indexes[s] = node_ind
            internal_nodes_to_process.pop()
        else:
            for mask in part_mask, anti_mask:
                if mask in TERMINAL_MASKS:
                    known_leaves_mask |= mask
                else:
                    # Add the new internal node's index to the new internal nodes stack, reusing old indexes
                    free_ind = internal_nodes_indexes.pop(0)  # Get the next free old index
                    internal_nodes_to_process.append((mask, free_ind))
    assert (not (internal_nodes_indexes))


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
    testset = normalize_points([perturbate_point(p, 0.0001) for p in random.sample(HAND, 1000)])
    # testset = normalize_points(HAND)
    # testset = normalize_points([perturbate_point(p, 0.000001) for p in random.sample(HAND, 10000)])
    # testset = normalize_points(HAND)
    # testset = HAND
    testpoints = [perturbate_point(p, 0.0) for p in random.sample(testset, 1000)]

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

    # print("construct morton-based binary tree")
    # start = timer()
    # morton_tree = construct_binary_tree_by_morton_codes(testset)
    # end = timer()
    # print("time to build ", (end - start))

    print("construct morton-based flat tree")
    start = timer()
    flat_tree = construct_flat_tree(testset)
    end = timer()
    print("time to build ", (end - start))


    # mortonized_points_sorted = sorted(mortonized_points)
    # m_point = mortonized_points_sorted[find_nearest_neighbor_morton(mortonized_points_sorted, testpoints[10])]
    # print(mortonized_points.index(m_point))
    start = timer()
    morton_results = []
    for test_point in testpoints[:1000]:
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
        print("")

    print ("Total search invocations:", SEARCH_INVOCATIONS)
    # start = timer()
    # for test_point in testpoints:
    #    found_point_linear = find_node_by_point_linear(testset, test_point)
    # end = timer()
    # print found_point_linear, dist(found_point_linear, test_point)
    # print (end - start)


if __name__ == "__main__":
    main()
