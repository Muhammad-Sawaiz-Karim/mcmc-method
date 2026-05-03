'''
Author: Muhammad Sawaiz Karim
Email: karim52@uwindsor.ca
Description: Problem II implemented from Rao, By A. Ramachandra et al. “A Markov chain Monte carol method for generating random (0, 1)-matrices with given marginals.” (1996).
Deals with the random, uniform generation of square matrices with structural zeroes.
'''

import random
import numpy as np
import itertools

def get_t_prob2(matrix):
    """Calculates t: minimum of 1s and NON-STRUCTURAL 0s."""
    n = matrix.shape[0]
    ones = np.sum(matrix == 1)
    
    # Total cells minus the diagonal, minus the 1s, gives non-structural 0s
    total_non_diagonal_cells = (n * n) - n
    non_structural_zeros = total_non_diagonal_cells - ones
    
    return min(ones, non_structural_zeros)

def find_all_cycles_prob2(matrix):
    """Finds alternating rectangles and compact alternating hexagons, ignoring the diagonal."""
    n = matrix.shape[0]
    cycles = set()
    
    # 1. Find Alternating Rectangles (excluding diagonals)
    for r1, r2 in itertools.combinations(range(n), 2):
        for c1, c2 in itertools.combinations(range(n), 2):
            # Skip if any coordinate hits the structural zero diagonal
            if r1 == c1 or r1 == c2 or r2 == c1 or r2 == c2:
                continue
                
            cells = ((r1, c1), (r1, c2), (r2, c1), (r2, c2))
            vals = [matrix[r, c] for r, c in cells]
            if vals == [1, 0, 0, 1] or vals == [0, 1, 1, 0]:
                cycles.add(frozenset(cells))
                
    # 2. Find Compact Alternating Hexagons
    # Defined by 3 distinct nodes: i1, i2, i3
    for i1, i2, i3 in itertools.permutations(range(n), 3):
        cells = ((i1, i2), (i1, i3), (i2, i3), (i2, i1), (i3, i1), (i3, i2))
        vals = [matrix[r, c] for r, c in cells]
        
        # Check if they alternate 1,0,1,0,1,0
        if vals == [1, 0, 1, 0, 1, 0] or vals == [0, 1, 0, 1, 0, 1]:
            cycles.add(frozenset(cells))
            
    return [list(c) for c in cycles]

def switch_along_cycle(matrix, cycle):
    """Flips the 0s and 1s at the specified cycle coordinates."""
    new_matrix = matrix.copy()
    for r, c in cycle:
        new_matrix[r, c] = 1 - new_matrix[r, c]
    return new_matrix

def generate_problem_2_matrices(initial_matrix, num_matrices_to_generate):
    """Generates random matrices for Problem II using MCMC Method III."""
    matrix = np.array(initial_matrix)
    t = get_t_prob2(matrix)
    
    max_used = 0
    mat_ct = 0
    run_count = 1
    r_pilot = 0
    
    results = []
    print(f"Generating {num_matrices_to_generate} random square matrices (Problem II)...")

    while mat_ct < num_matrices_to_generate:
        A = matrix.copy()
        step = 0
        
        if run_count == 1:
            cycles = find_all_cycles_prob2(A)
            acct = len(cycles)
            r_pilot = 2 * acct
            
        while step < 3 * t:
            cycles = find_all_cycles_prob2(A)
            acct = len(cycles)
            max_used = max(max_used, acct)
            
            if max_used > 0:
                ird = random.randint(1, max_used)
                if ird <= acct:
                    A = switch_along_cycle(A, cycles[ird - 1])
                    
            step += 1
            
        if run_count > r_pilot:
            results.append(A)
            mat_ct += 1
        elif run_count == r_pilot:
            max_used = (10 * max_used // 9) + 1
            
        run_count += 1
        
    return results

if __name__ == "__main__":
    # Example 4x4 Social Network (0s down the diagonal)
    initial_A = [
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ]
    sample = generate_problem_2_matrices(initial_A, 12)
    for idx, mat in enumerate(sample):
        print(f"\nRandom Matrix {idx + 1}:\n{mat}")
