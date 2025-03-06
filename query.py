from typing import Iterable, Optional

import dforest

import numpy as np

def max_query(forest: dforest.Forest, min_bound: Iterable[float],
              max_bound: Iterable[float], threshold: float,
              n_samples: int = 10_000, batch_size: int = 10_000,
              merge_limit: Optional[int] = None,
              verbose: bool = False) -> Optional[bool]:
    if merge_limit is None:
        merge_limit = forest.n_trees / 4
    forest = forest.copy()
    forest.prune_box(min_bound, max_bound)

    def check_query():
        if forest.max_bound is not None and forest.max_bound <= threshold:
            return True
        if forest.champ_max is not None and forest.champ_max > threshold:
            return False
        return None

    def exit_query():
        if forest.max_bound is not None and forest.max_bound <= threshold:
            if verbose:
                print("Query holds")
                print(f"f(x) <= {threshold}")
                print(f"{forest.n_trees} trees, max bound: {forest.max_bound}")
            forest.free()
            return True
        if forest.champ_max is not None and forest.champ_max > threshold:
            if verbose:
                print("Query disproven")
                print(f"f(x) > {threshold}")
                print(f"x = {forest.champ_max_x}, f(x) = {forest.champ_max}")
            forest.free()
            return False
        return None
    
    if check_query() is not None:
        return exit_query()
    
    n_tested = 0
    while n_tested < n_samples:
        forest.sample(min_bound, max_bound, batch_size)
        n_tested += batch_size

        if check_query() is not None:
            return exit_query()

    if check_query() is not None:
        return exit_query()
    
    while forest.n_trees > merge_limit:
        forest.merge(forest.n_trees - 1)

        if check_query() is not None:
            return exit_query()

    print(f"Query failed. Cannot (dis)prove f(x) <= {threshold}.")
    forest.print_summary()
    forest.free()
    return None
