import time

import query

def run_robustness_evals(model, samples, delta, eps, hyperparams, output=True):
    results = []
    times = []
    N = len(samples)

    for i, param in enumerate(hyperparams):
        start_time = time.time()
        res = query.robustness_query_many(model, samples, delta, eps, **param)
        elapsed_time = time.time() - start_time

        results.append(res)
        times.append(elapsed_time)

        if output:
            print(f"Batch {i + 1}")
            print(f"True: {len(res[0])}")
            print(f"False: {len(res[1])}")
            print(f"None: {len(res[2])}")
            print(f"Time batch {i + 1}: {elapsed_time:.4f}")
            print(f"{(elapsed_time / len(samples)):.4f} s / sample")
            print()
        
        samples = res[2]
    
    if output:
        true = [len(x[0]) for x in results]
        false = [len(x[1]) for x in results]
        none = len(results[-1][2])

        print("Total")
        print(f"True: {sum(true)}")
        print(f"False: {sum(false)}")
        print(f"None: {none}")
        print(f"Time: {sum(times):.4f}")
        print(f"{(sum(times) / N):.4f} s / sample")
        print()
    
    return results, time
