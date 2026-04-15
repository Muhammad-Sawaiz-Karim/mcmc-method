import random
import numpy as np
import itertools

def get_t(matrix):
    """Calculates t: the minimum of the number of 1s and 0s."""
    ones = np.sum(matrix == 1)
    zeros = np.sum(matrix == 0)
    return min(ones, zeros)

def find_alternating_rectangles(matrix):
    """Finds all alternating rectangles (1,0,0,1 or 0,1,1,0)."""
    rectangles = set()
    rows, cols = matrix.shape
    
    # Check every possible combination of 2 rows and 2 columns
    for r1, r2 in itertools.combinations(range(rows), 2):
        for c1, c2 in itertools.combinations(range(cols), 2):
            cells = ((r1, c1), (r1, c2), (r2, c1), (r2, c2))
            vals = [matrix[r, c] for r, c in cells]
            
            # If the corners alternate, save the coordinates
            if vals == [1, 0, 0, 1] or vals == [0, 1, 1, 0]:
                # Use frozenset to prevent duplicate counting
                rectangles.add(frozenset(cells))
                
    return [list(r) for r in rectangles]

def switch_along_cycle(matrix, cycle):
    """Flips the 0s and 1s at the specified cycle coordinates."""
    new_matrix = matrix.copy()
    for r, c in cycle:
        new_matrix[r, c] = 1 - new_matrix[r, c]
    return new_matrix

def generate_problem_1_matrices(initial_matrix, num_matrices_to_generate):
    """Generates random matrices for Problem I using MCMC Method III."""
    matrix = np.array(initial_matrix)
    t = get_t(matrix)
    
    max_used = 0
    mat_ct = 0
    run_count = 1
    r_pilot = 0
    
    results = []
    print(f"Generating {num_matrices_to_generate} random matrices (Problem I)...")

    while mat_ct < num_matrices_to_generate:
        A = matrix.copy()
        step = 0
        
        # Determine pilot phase length on the first run
        if run_count == 1:
            cycles = find_alternating_rectangles(A)
            acct = len(cycles)
            r_pilot = 2 * acct
            
        while step < 3 * t:
            cycles = find_alternating_rectangles(A)
            acct = len(cycles)
            max_used = max(max_used, acct)
            
            if max_used > 0:
                ird = random.randint(1, max_used)
                if ird <= acct:
                    A = switch_along_cycle(A, cycles[ird - 1])
                    
            step += 1
            
        # Save matrices after pilot phase
        if run_count > r_pilot:
            results.append(A)
            mat_ct += 1
        elif run_count == r_pilot:
            # Pilot complete: Inflate max_used to set safe upper bound K
            max_used = (10 * max_used // 9) + 1
            
        run_count += 1
        
    return results

if __name__ == "__main__":
    # Example 3x4 Matrix
    initial_A = [
        [1, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ]
    sample = generate_problem_1_matrices(initial_A, 3)
    for idx, mat in enumerate(sample):
        print(f"\nRandom Matrix {idx + 1}:\n{mat}")