import time

import query
import smt_query

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

def run_smt_evals(model, samples, delta, eps, gb=False, timeout=300,
                  output=None):
    N = len(samples)

    start_time = time.time()
    res = smt_query.query_many(model, samples, delta, eps, gb=gb,
                               timeout=timeout)
    elapsed_time = time.time() - start_time
    
    if output is not None:
        with open(output, "a") as file:

            file.write("Total\n")
            file.write(f"True: {len(res[0])}\n")
            file.write(f"False: {len(res[1])}\n")
            file.write(f"None: {len(res[2])}\n")
            file.write(f"Time: {elapsed_time:.4f}\n")
            file.write(f"{(elapsed_time / N):.4f} s / sample\n")
    
    return res, elapsed_time

def run_multiclass_evals(model, samples, delta, hyperparams,
                         clip_min=None, clip_max=None, output=None):
    results = []
    cexs = []
    times = []
    N = len(samples)

    for i, param in enumerate(hyperparams):
        if len(samples) == 0:
            break
        start_time = time.time()
        res, cex = query.multiclass_rob_query_many(model, samples, delta,
                                              clip_min, clip_max, **param)
        elapsed_time = time.time() - start_time

        results.append(res)
        cexs.append(cex)
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
    
    return results, cexs, times
