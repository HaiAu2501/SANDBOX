import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from concurrent.futures import ThreadPoolExecutor
from aco import run_tsp_aco

OPTIMAL = {
    "50": 5.706510615938985,
    "100": 7.7894944810761135, 
    "200": 10.678417491907155,
    "300": 13.002009389982788,
    "500": 16.547266843098683
}

def eval_instance(coords, n_ants, n_iter, seed):
    D = distance_matrix(coords, coords)
    return run_tsp_aco(D, n_ants, n_iter, seed)

def process_file(path, n_ants, n_iter):
    data = np.load(path)  # shape (n_instances, size, 2)
    n_instances = data.shape[0]
    seeds = np.arange(n_instances)
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(eval_instance, data[i], n_ants, n_iter, int(seeds[i]))
            for i in range(n_instances)
        ]
        for f in futures:
            results.append(f.result())
    return np.array(results)

def main(size):
    print(f"Processing TSP instances of size {size}...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'datasets', f'test_TSP{size}.npy')
    list_costs = process_file(path, n_ants=50, n_iter=200)
    mean_cost = np.mean(list_costs, axis=0).tolist() # Average costs over all instances
    best = mean_cost[-1]
    optimal = OPTIMAL[str(size)]
    opt_gap = (best - optimal) / optimal * 100
    print(f"Optimal gap for TSP{size}: {opt_gap:.2f}%")

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(size)
