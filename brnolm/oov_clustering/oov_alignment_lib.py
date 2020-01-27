import numpy as np
import copy


def word_distance(a, b):
    return 0 if a == b else 1


DIAGONAL_MOVE = 0
HORIZONAL_MOVE = 1
VERTICAL_MOVE = -1


def path_from_moves(moves_taken):
    ptr_a = moves_taken.shape[0] - 1
    ptr_b = moves_taken.shape[1] - 1

    path = []
    while ptr_a != 0 or ptr_b != 0:
        move = int(moves_taken[ptr_a, ptr_b])
        path.append(move)

        if move == VERTICAL_MOVE:
            ptr_a -= 1
        elif move == HORIZONAL_MOVE:
            ptr_b -= 1
        else:
            ptr_a -= 1
            ptr_b -= 1

    return list(reversed(path))


def ind_ali_from_path(path):
    alignment = []
    ptr_a = 0
    ptr_b = 0
    inds_a = []
    inds_b = []
    for move in path:
        if move == VERTICAL_MOVE:
            inds_a.append(ptr_a)
            ptr_a += 1
        elif move == HORIZONAL_MOVE:
            inds_b.append(ptr_b)
            ptr_b += 1
        else:
            inds_a.append(ptr_a)
            inds_b.append(ptr_b)
            ptr_a += 1
            ptr_b += 1

            alignment.append((inds_a, inds_b))
            inds_a = []
            inds_b = []

    if len(inds_a) > 0 or len(inds_b) > 0:
        assert(len(inds_a) == 0 or len(inds_b) == 0)
        alignment[-1] = (
            alignment[-1][0] + inds_a,
            alignment[-1][1] + inds_b,
        )

    return alignment


def local_costs_from_strings(a, b):
    local_costs = np.zeros(shape=(len(a)+1, len(b)+1))

    for i, w_a in enumerate(a):
        for j, w_b in enumerate(b):
            local_costs[i+1, j+1] = word_distance(w_a, w_b)

    return local_costs


def path_from_local_costs(local_costs):
    partial_costs = np.full(shape=local_costs.shape, fill_value=np.inf)
    moves_taken = np.full(shape=local_costs.shape, fill_value=np.inf)

    for i in range(partial_costs.shape[0]):
        for j in range(partial_costs.shape[1]):
            if i == 0 and j == 0:
                partial_costs[i, j] = 0.0
                moves_taken[i, j] = DIAGONAL_MOVE
            elif i == 0 and j != 0:
                partial_costs[i, j] = partial_costs[i, j-1] + 1.0
                moves_taken[i, j] = HORIZONAL_MOVE
            elif i != 0 and j == 0:
                partial_costs[i, j] = partial_costs[i-1, j] + 1.0
                moves_taken[i, j] = VERTICAL_MOVE
            else:
                vertical_cost = partial_costs[i-1, j] + 1.0
                horizontal_cost = partial_costs[i, j-1] + 1.0
                diagonal_cost = partial_costs[i-1, j-1] + local_costs[i, j]

                best_cost = min([vertical_cost, horizontal_cost, diagonal_cost])

                partial_costs[i, j] = best_cost
                if best_cost == diagonal_cost:
                    moves_taken[i, j] = DIAGONAL_MOVE
                elif best_cost == horizontal_cost:
                    moves_taken[i, j] = HORIZONAL_MOVE
                else:
                    moves_taken[i, j] = VERTICAL_MOVE

    return moves_taken


def word_ali_from_index_ali(a, b, index_alignment):
    word_alignment = []
    for inds_a, inds_b in index_alignment:
        word_alignment.append((
            [a[ind_a] for ind_a in inds_a],
            [b[ind_b] for ind_b in inds_b],
        ))

    return word_alignment


def align(a, b):
    local_costs = local_costs_from_strings(a, b)
    moves_taken = path_from_local_costs(local_costs)
    path = path_from_moves(moves_taken)
    index_alignment = ind_ali_from_path(path)
    return word_ali_from_index_ali(a, b, index_alignment)


def insertion_mismatch(a, b):
    assert(len(a) < len(b))
    assert(len(a) == 1)

    if a[0] == b[0]:
        mismatch = ([], b[1:])
        ends_with_mismatch = True
    elif a[0] == b[-1]:
        mismatch = ([], b[:-1])
        ends_with_mismatch = False
    else:
        mismatch = (a, b)
        ends_with_mismatch = True

    return mismatch, ends_with_mismatch


def equal_lenght_mismatch(a, b):
    assert(len(a) == 1 and len(b) == 1)
    if a != b:
        mismatch = (a, b)
        ends_with_mismatch = True
    else:
        mismatch = None
        ends_with_mismatch = False

    return mismatch, ends_with_mismatch


def single_pair_mismatch(a, b):
    if len(a) == len(b):
        mismatch, ends_with_mismatch = equal_lenght_mismatch(a, b)
    elif len(a) < len(b):
        mismatch, ends_with_mismatch = insertion_mismatch(a, b)
    elif len(a) > len(b):
        mismatch, ends_with_mismatch = insertion_mismatch(b, a)
        mismatch = tuple(reversed(mismatch))

    return mismatch, ends_with_mismatch


def extract_mismatch(ali):
    mismatches = []
    last_was_mismatched = False
    for a, b in ali:
        do_extend = copy.deepcopy(last_was_mismatched)
        mismatch, last_was_mismatched = single_pair_mismatch(a, b)

        if not mismatch:
            continue

        if do_extend:
            mismatches[-1][0].extend(mismatch[0])
            mismatches[-1][1].extend(mismatch[1])
        else:
            mismatches.append(mismatch)

    return mismatches


def find_in_mismatches(mismatches, word):
    for m in mismatches:
        if word in m[0] or word in m[1]:
            return m


def number_of_errors(mismatches):
    nb_errors = 0
    for a, b in mismatches:
        nb_errors += 1 + abs(len(a) - len(b))
    return nb_errors
