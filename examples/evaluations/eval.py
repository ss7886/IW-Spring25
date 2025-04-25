import time

import query

def run_robustness_evals(model, samples, delta, eps, hyperparams, output=None):
    results = []
    times = []
    N = len(samples)

    for i, param in enumerate(hyperparams):
        if len(samples) == 0:
            break
        start_time = time.time()
        res = query.robustness_query_many(model, samples, delta, eps, **param)
        elapsed_time = time.time() - start_time

        results.append(res)
        times.append(elapsed_time)

        if output is not None:
            with open(output, "a") as file:
                file.write(f"Batch {i + 1}\n")
                file.write(f"True: {len(res[0])}\n")
                file.write(f"False: {len(res[1])}\n")
                file.write(f"None: {len(res[2])}\n")
                file.write(f"Time batch {i + 1}: {elapsed_time:.4f}\n")
                file.write(f"{(elapsed_time / len(samples)):.4f} s / sample\n\n")
        
        samples = res[2]
    
    if output is not None:
        with open(output, "a") as file:
            true = [len(x[0]) for x in results]
            false = [len(x[1]) for x in results]
            none = len(results[-1][2])

            file.write("Total\n")
            file.write(f"True: {sum(true)}\n")
            file.write(f"False: {sum(false)}\n")
            file.write(f"None: {none}\n")
            file.write(f"Time: {sum(times):.4f}\n")
            file.write(f"{(sum(times) / N):.4f} s / sample\n")
    
    return results, times
