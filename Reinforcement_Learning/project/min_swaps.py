from collections import deque
from operator import itemgetter
from itertools import groupby

def min_swaps(sequence):
    """
    Returns miminum number of transpositions/swaps to sort given sequence.

    Much faster than a modified (and even Cython-optimized) implementation
    of selection sort, even for short arrays.
    
    This is based on the finding disjoint permutation cycles between
    the sequence and its sorted order. It takes exactly length(cycle)-1
    transpositions to reverse a cycle.
    """

    # do lexical sort on (value, original_index) pairs
    # however sort is stable, so following is an option too (pairs are (original_index, value))
    value_getter = itemgetter(1)
    sorted_idcs_vals = sorted(enumerate(sequence), key=value_getter)
    # get relevant indices for every value
    val_to_idcs = {
        value:deque(
            idx for idx, _ in indices
            # if element is not stationary
            if sequence[idx]!=sorted_idcs_vals[idx][1]
        )
        for value, indices in groupby(sorted_idcs_vals, key=value_getter)
    }
    
    # The permutaion cycles can be built by uncommenting lines with "cycle(s)" 
    # I keep it here - it clarifies the algorithm and might be useful some other day
    #cycles = []
    swaps = 0
    for start_value in sequence:
        indices = val_to_idcs[start_value]

        # we have seen the element(s) already or it was stationary
        if not indices: continue

        idx = indices.popleft()
        next_value = sorted_idcs_vals[idx][1]

        #cycle = [idx]
        while start_value != next_value:
            next_idx = val_to_idcs[next_value].popleft()
            next_value = sorted_idcs_vals[next_idx][1]
            #cycle.append(next_idx)
            swaps += 1

        #cycles.append(cycle)
        
    return swaps #, cycles

## modified selection sort in Cython
# import numpy as np
# cimport numpy as np
# cpdef int min_swapss(np.ndarray[np.int_t, ndim=1] a, int n):
#     cdef int swaps = 0
#     cdef int i, min_i
#     for i in range(n-1):
#         min_i = np.argmin(a[i:])
#         if min_i != i:
#             a[min_i] = a[i]
#             swaps += 1
#     return swaps


# def min_swaps(sequence):
#     original_vals = sequence
#     length = len(original_vals)
#     value_getter = operator.itemgetter(1)
#     sorted_idcs_vals = sorted(enumerate(original_vals), key=value_getter)
#     #print(f'ov: {original_vals}\nsiv: {sorted_idcs_vals}')
#     available = [original_vals[idx]!=sorted_idcs_vals[idx][1] for idx in range(length)]
#     val_to_idx = {
#         k:(list(
#             idx for idx, _ in g
#             if available[idx]
#             #if original_vals[idx]!=sorted_idcs_vals[idx][1]
#         ))
#         for k, g in it.groupby(sorted_idcs_vals, key=value_getter)
#     }
    
    

#     permutations = []
#     swaps = 0
#     for i in range(length):

#         if not available[i]: continue

#         start_val = original_vals[i]
#         target_val = sorted_idcs_vals[i][1]

#         permutation = [i]

#         while start_val != target_val:
#             #ni = next(val_to_idx[target_val])
#             #ni = val_to_idx[target_val].popleft()
#             ni = val_to_idx[target_val].pop(0)
#             available[ni] = False
#             target_val = sorted_idcs_vals[ni][1]
#             permutation.append(ni)
#             swaps += 1

#         val_to_idx[target_val].pop(0)
#         available[i] = False
#         permutations.append(permutation)
        
#     return swaps, permutations

if __name__=="__main__":
    arr = [3,2,2,2,4,3,2,1,6,5,7,4,6,6,5,4,8,4,4,1,1,3,2,2,2,2,6,2,2,3,5,8,4,2,2]
    a = arr[:]
    a = [2,2,2,2,2,2,1,1,1,1,1,1]
    print(a)
    ms = min_swaps(a)
    print(ms)