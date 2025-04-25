from typing import Iterable, Optional, Tuple, Union, TYPE_CHECKING

from dforest import Forest
from dforest_class import ForestClassifier

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    Vector = Union[Iterable[float], NDArray]
else:
    Vector = object

def pso_max(forest: Forest, min_bound: Vector, max_bound: Vector,
            N: int = 10_000, forest_copy: bool = True, max_iters: int = 20,
            inertia: float = 0.8, part_inf: float = 1, swarm_inf: float = 1,
            stop: Optional[float] = None) -> tuple[NDArray, float]:
    assert len(min_bound) == forest.dim
    assert len(max_bound) == forest.dim
    assert max_iters > 0
    assert 0 <= inertia < 1

    if forest_copy:
        forest = forest.copy().prune_box(min_bound, max_bound)

    swarm = np.random.uniform(min_bound, max_bound, (N, forest.dim))
    velocity = np.zeros((N, forest.dim))
    particle_best = np.zeros((N, forest.dim))
    particle_best_y = np.ones(N) * -np.inf
    swarm_best = np.zeros(forest.dim)
    swarm_best_y = -np.inf

    for t in range(max_iters):
        # Eval
        y = forest.eval(swarm)
        for i, particle in enumerate(swarm):
            if y[i] > particle_best_y[i]:
                particle_best[i] = particle.copy()
                particle_best_y[i] = y[i]
            if y[i] > swarm_best_y:
                swarm_best = particle.copy()
                swarm_best_y = y[i]
        
        if stop is not None and swarm_best_y > stop:
            break
        
        # Calculate velocity
        velocity *= inertia
        r1, r2 = np.random.uniform(0, 1, N), np.random.uniform(0, 1, N)
        velocity += part_inf * np.expand_dims(r1, axis=1) * (particle_best - swarm)
        velocity += swarm_inf * np.expand_dims(r2, axis=1) * -(swarm - swarm_best)

        # Update positions
        swarm += velocity
        np.clip(swarm, min_bound, max_bound, swarm)
    
    if forest_copy:
        forest.free()
    
    return swarm_best, swarm_best_y

def pso_min(forest: Forest, min_bound: Vector, max_bound: Vector,
            N: int = 10_000, forest_copy: bool = True, max_iters: int = 20,
            inertia: float = 0.8, part_inf: float = 1, swarm_inf: float = 1,
            stop: Optional[float] = None) -> tuple[NDArray, float]:
    assert len(min_bound) == forest.dim
    assert len(max_bound) == forest.dim
    assert max_iters > 0
    assert 0 <= inertia < 1

    if forest_copy:
        forest = forest.copy().prune_box(min_bound, max_bound)

    swarm = np.random.uniform(min_bound, max_bound, (N, forest.dim))
    velocity = np.zeros((N, forest.dim))
    particle_best = np.zeros((N, forest.dim))
    particle_best_y = np.ones(N) * np.inf
    swarm_best = np.zeros(forest.dim)
    swarm_best_y = np.inf

    for t in range(max_iters):
        # Eval
        y = forest.eval(swarm)
        for i, particle in enumerate(swarm):
            if y[i] < particle_best_y[i]:
                particle_best[i] = particle.copy()
                particle_best_y[i] = y[i]
            if y[i] < swarm_best_y:
                swarm_best = particle.copy()
                swarm_best_y = y[i]
        
        if stop is not None and swarm_best_y < stop:
            break
        
        # Calculate velocity
        velocity *= inertia
        r1, r2 = np.random.uniform(0, 1, N), np.random.uniform(0, 1, N)
        velocity += part_inf * np.expand_dims(r1, axis=1) * (particle_best - swarm)
        velocity += swarm_inf * np.expand_dims(r2, axis=1) * -(swarm - swarm_best)

        # Update positions
        swarm += velocity
        np.clip(swarm, min_bound, max_bound, swarm)
    
    if forest_copy:
        forest.free()
    
    return swarm_best, swarm_best_y

def max_query(forest: Forest, min_bound: Vector,
              max_bound: Vector, threshold: float,
              pso_N: int = 10_000, pso_max_iters: int = 10,
              merge_limit: Optional[int] = None,
              branch_and_bound: bool = True, offset: float = 0.0,
              prune: bool = True, verbose: bool = False) -> Optional[bool]:
    if merge_limit is None:
        merge_limit = forest.n_trees / 4
    
    forest = forest.copy()
    if prune:
        forest.prune_box(min_bound, max_bound)

    # Check if query has been (dis)proven, returns None otherwise
    def check_query():
        if forest.max_bound is not None and forest.max_bound <= threshold:
            return True
        if forest.champ_max is not None and forest.champ_max > threshold:
            return False
        return None

    # Exit function
    def exit_query():
        if forest.max_bound is not None and forest.max_bound <= threshold:
            if verbose:
                print(f"Query f(x) <= {threshold} holds")
                print(f"{forest.n_trees} trees, max bound: {forest.max_bound}")
            forest.free()
            return True
        if forest.champ_max is not None and forest.champ_max > threshold:
            if verbose:
                print(f"Query f(x) <= {threshold} disproven")
                print(f"x = {forest.champ_max_x}, f(x) = {forest.champ_max}")
            forest.free()
            return False
        if verbose:
            print("Can not (dis)prove query.")
            forest.print_summary()
        forest.free()
        return None
    
    if check_query() is not None:
        return exit_query()
    
    # Run pso (to find counter examples, raise lower bound)
    best_x, _ = pso_max(forest, min_bound, max_bound, N=pso_N, 
                        forest_copy=False, max_iters=pso_max_iters,
                        stop=threshold)

    if check_query() is not None:
        return exit_query()
    
    # Merge trees (to decrease upper bound)
    while forest.n_trees > merge_limit:
        if branch_and_bound:
            forest.merge_max(forest.n_trees - 1, best_x, offset=offset)
        else:
            forest.merge(forest.n_trees - 1)

        if check_query() is not None:
            return exit_query()
    
    # Try to find max
    for tree in forest:
        opt_min_bound, opt_max_bound = tree.find_max(min_bound.copy(),
                                                     max_bound.copy())
        opt_x = (opt_min_bound + opt_max_bound) / 2
        forest.eval(opt_x)

    return exit_query()

def min_query(forest: Forest, min_bound: Vector,
              max_bound: Vector, threshold: float,
              pso_N: int = 10_000, pso_max_iters: int = 10,
              merge_limit: Optional[int] = None,
              branch_and_bound: bool = True, offset: float = 0.0,
              prune: bool = True, verbose: bool = False) -> Optional[bool]:
    if merge_limit is None:
        merge_limit = forest.n_trees / 4
    
    forest = forest.copy()
    if prune:
        forest.prune_box(min_bound, max_bound)

    # Check if query has been (dis)proven, returns None otherwise
    def check_query():
        if forest.min_bound is not None and forest.min_bound >= threshold:
            return True
        if forest.champ_min is not None and forest.champ_min < threshold:
            return False
        return None

    # Exit function
    def exit_query():
        if forest.min_bound is not None and forest.min_bound >= threshold:
            if verbose:
                print(f"Query f(x) >= {threshold} holds")
                print(f"{forest.n_trees} trees, min bound: {forest.min_bound}")
            forest.free()
            return True
        if forest.champ_min is not None and forest.champ_min < threshold:
            if verbose:
                print(f"Query f(x) >= {threshold} disproven")
                print(f"x = {forest.champ_min_x}, f(x) = {forest.champ_min}")
            forest.free()
            return False
        if verbose:
            print("Can not (dis)prove query.")
            print()
        forest.free()
        return None
    
    if check_query() is not None:
        return exit_query()
    
    # Run pso (to find counter examples, raise lower bound)
    best_x, _ = pso_min(forest, min_bound, max_bound, N=pso_N,
                        forest_copy=False, max_iters=pso_max_iters,
                        stop=threshold)

    if check_query() is not None:
        return exit_query()
    
    # Merge trees (to decrease upper bound)
    while forest.n_trees > merge_limit:
        if branch_and_bound:
            forest.merge_min(forest.n_trees - 1, best_x, offset=offset)
        else:
            forest.merge(forest.n_trees - 1)

        if check_query() is not None:
            return exit_query()
    
    for tree in forest:
        opt_min_bound, opt_max_bound = tree.find_min(min_bound.copy(),
                                                     max_bound.copy())
        opt_x = (opt_min_bound + opt_max_bound) / 2
        forest.eval(opt_x)

    return exit_query()

def query(forest: Forest, min_bound: Vector, max_bound: Vector,
          min_threshold: float, max_threshold: float,
          offset_factor: float = 0.15, **kwargs) -> Optional[bool]:
    forest = forest.copy()
    forest.prune_box(min_bound, max_bound)

    offset = offset_factor * (max_threshold - min_threshold)
    min_result = min_query(forest, min_bound, max_bound, min_threshold,
                           offset=offset, prune=False, **kwargs)
    
    if min_result == False:
        forest.free()
        return False
    
    offset *= -1
    max_result = max_query(forest, min_bound, max_bound, max_threshold,
                           offset=offset, prune=False, **kwargs)
    
    forest.free()
    if min_result and max_result:
        return True
    if max_result == False:
        return False
    return None

def robustness_query(forest: Forest, x: Vector, delta: Vector,
                     epsilon: float, **kwargs) -> Optional[bool]:
    """
    Assume f(x) is forest's prediction given x as an input.

    Attempts to prove the following robustness query or provide a counter-
    example:

        Forall x', such that x - delta <= x' <= x + delta, show that
        f(x) - epsilon <= f(x') <= f(x) + epsilon.
    """
    y = forest.eval(x)
    return query(forest, x - delta, x + delta, y - epsilon, y + epsilon,
                 **kwargs)

def robustness_query_many(forest: Forest, X: Iterable[Vector], delta: Vector,
                          epsilon: float, **kwargs) -> Tuple[Iterable[Vector],
                                                             Iterable[Vector],
                                                             Iterable[Vector]]:
    true_x = []
    false_x = []
    unproven = []
    for x in X:
        result = robustness_query(forest, x, delta, epsilon, **kwargs)
        if result is None:
            unproven.append(x.copy())
        elif result:
            true_x.append(x.copy())
        else:
            false_x.append(x.copy())
    return true_x, false_x, unproven

def pso_two_class(forest: ForestClassifier, class_a: int, class_b: int,
                  min_bound: Vector, max_bound: Vector, N: int = 10_000,  
                  forest_copy: bool = True, max_iters: int = 20,
                  inertia: float = 0.8, part_inf: float = 1,
                  swarm_inf: float = 1,
                  stop: Optional[float] = 0) -> tuple[NDArray, float]:
    assert len(min_bound) == forest.dim
    assert len(max_bound) == forest.dim
    assert max_iters > 0
    assert 0 <= inertia < 1

    if forest_copy:
        forest = forest.copy()
        forest.prune_box(min_bound, max_bound, class_ids=[class_a, class_b])

    swarm = np.random.uniform(min_bound, max_bound, (N, forest.dim))
    velocity = np.zeros((N, forest.dim))
    particle_best = np.zeros((N, forest.dim))
    particle_best_y = np.ones(N) * -np.inf
    swarm_best = np.zeros(forest.dim)
    swarm_best_y = -np.inf

    for t in range(max_iters):
        # Eval
        y = forest.eval(swarm, probs=True, raw=forest._gb,
                        class_ids=[class_a, class_b])
        y = y[:, 1] - y[:, 0]
        for i, particle in enumerate(swarm):
            if y[i] > particle_best_y[i]:
                particle_best[i] = particle.copy()
                particle_best_y[i] = y[i]
            if y[i] > swarm_best_y:
                swarm_best = particle.copy()
                swarm_best_y = y[i]
        
        if stop is not None and swarm_best_y > stop:
            break
        
        # Calculate velocity
        velocity *= inertia
        r1, r2 = np.random.uniform(0, 1, N), np.random.uniform(0, 1, N)
        velocity += part_inf * np.expand_dims(r1, axis=1) * (particle_best - swarm)
        velocity += swarm_inf * np.expand_dims(r2, axis=1) * -(swarm - swarm_best)

        # Update positions
        swarm += velocity
        np.clip(swarm, min_bound, max_bound, swarm)
    
    if forest_copy:
        forest.free()
    
    return swarm_best, swarm_best_y

def pso_one_class(forest: ForestClassifier, class_a: int, min_bound: Vector,
                  max_bound: Vector, N: int = 10_000, forest_copy: bool = True,
                  max_iters: int = 20, inertia: float = 0.8,
                  part_inf: float = 1, swarm_inf: float = 1,
                  stop: Optional[float] = None) -> tuple[NDArray, float]:
    assert len(min_bound) == forest.dim
    assert len(max_bound) == forest.dim
    assert max_iters > 0
    assert 0 <= inertia < 1

    if forest_copy:
        forest = forest.copy()
        forest.prune_box(min_bound, max_bound)
    
    other_cols = list(range(forest.n_class))
    other_cols.pop(class_a)

    swarm = np.random.uniform(min_bound, max_bound, (N, forest.dim))
    velocity = np.zeros((N, forest.dim))
    particle_best = np.zeros((N, forest.dim))
    particle_best_y = np.ones(N) * -np.inf
    swarm_best = np.zeros(forest.dim)
    swarm_best_y = -np.inf

    for t in range(max_iters):
        # Eval
        y = forest.eval(swarm, probs=True, raw=forest._gb)
        y = y[:, class_a] - np.max(y[:, other_cols], axis=1)
        for i, particle in enumerate(swarm):
            if y[i] > particle_best_y[i]:
                particle_best[i] = particle.copy()
                particle_best_y[i] = y[i]
            if y[i] > swarm_best_y:
                swarm_best = particle.copy()
                swarm_best_y = y[i]
        
        if stop is not None and swarm_best_y > stop:
            break
        
        # Calculate velocity
        velocity *= inertia
        r1, r2 = np.random.uniform(0, 1, N), np.random.uniform(0, 1, N)
        velocity += part_inf * np.expand_dims(r1, axis=1) * (particle_best - swarm)
        velocity += swarm_inf * np.expand_dims(r2, axis=1) * -(swarm - swarm_best)

        # Update positions
        swarm += velocity
        np.clip(swarm, min_bound, max_bound, swarm)
    
    if forest_copy:
        forest.free()
    
    return swarm_best, swarm_best_y

def two_class_query(forest: ForestClassifier, class_a: int, class_b: int,
                    min_bound: Vector, max_bound: Vector, prune: bool = True,
                    pso_N: int = 10_000, pso_max_iters: int = 10,
                    merge_limit: Optional[int] = None,
                    branch_and_bound: bool = True, offset: float = 0.0,
                    verbose: bool = False) -> Tuple[Optional[bool],
                                                    Optional[Vector]]:
    if merge_limit is None:
        merge_limit = forest.n_trees[class_a] / 4

    forest = forest.copy()
    if prune:
        forest.prune_box(min_bound, max_bound, class_ids=[class_a, class_b])

    best_x = None
    best_diff = -1

    # Check if query has been (dis)proven, returns None otherwise
    def check_query():
        if forest.min_bound[class_a] > forest.max_bound[class_b]:
            return True
        if best_diff > 0:
            return False
        return None

    # Exit function
    def exit_query():
        label_a, label_b = forest.classes[class_a], forest.classes[class_b]
        if forest.min_bound[class_a] > forest.max_bound[class_b]:
            if verbose:
                print(f"Query holds for class {label_a} over class {label_b}")
                print(f"{forest.n_trees} trees, min bound: {forest.min_bound}")
            forest.free()
            return True, None
        if best_diff > 0:
            if verbose:
                print(f"Query disproven for class {label_a}")
                print(f"x = {best_x}, y = {forest.eval(best_x)}")
            forest.free()
            return False, best_x
        return None, None
    
    if check_query() is not None:
        return exit_query()
    
    best_x, best_diff = pso_two_class(forest, class_a, class_b, min_bound,
                                      max_bound, N=pso_N,
                                      max_iters=pso_max_iters)
    
    if check_query() is not None:
        return exit_query()
    
    while forest.n_trees[class_a] > merge_limit:
        n = forest.n_trees[class_a] - 1
        if branch_and_bound:
            forest.merge_min(class_a, n, best_x, offset)
            forest.merge_max(class_b, n, best_x, -offset)
        else:
            forest.merge(class_a, n)
            forest.merge(class_b, n)

        if check_query() is not None:
            return exit_query()
    
    forest.free()

    return exit_query()

def multiclass_query(forest: ForestClassifier, x: Vector,
                     min_bound: Vector, max_bound: Vector,
                     **kwargs) -> Tuple[Optional[bool], Optional[Vector]]:
    forest = forest.copy()
    forest.prune_box(min_bound, max_bound)

    y = np.argmax(forest.eval(x, True)[0])

    res = True
    for i in range(forest.n_class):
        if i == y:
            continue
        query_res, cex = two_class_query(forest, y, i, min_bound, max_bound,
                                         prune=False, **kwargs)
        if query_res == False:
            return False, cex
        elif query_res is None:
            res = None
    return res, None

def multiclass_robustness_query(forest: ForestClassifier, x: Vector,
                                delta: Vector, clip_min: Vector = None,
                                clip_max: Vector = None,
                                **kwargs) -> Tuple[Optional[bool],
                                                   Optional[Vector]]:
    min_bound = np.clip(x - delta, clip_min, clip_max)
    max_bound = np.clip(x + delta, clip_min, clip_max)
    return multiclass_query(forest, x, min_bound, max_bound, **kwargs)

def multiclass_robustness_query_many(forest: Forest, X: Iterable[Vector],
                                     delta: Vector, clip_min: Vector = None,
                                     clip_max: Vector = None,
                                     **kwargs) -> Tuple[Tuple[Iterable[Vector],
                                                              Iterable[Vector],
                                                              Iterable[Vector]],
                                                        Iterable[Vector]]:
    true_x = []
    false_x = []
    unproven = []
    cexs = []
    for x in X:
        result, cex = multiclass_robustness_query(forest, x, delta, clip_min,
                                                  clip_max, **kwargs)
        if result is None:
            unproven.append(x.copy())
        elif result:
            true_x.append(x.copy())
        else:
            false_x.append(x.copy())
            cexs.append(cex)

    return (true_x, false_x, unproven), cexs
