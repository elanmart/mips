def evalaute(D, I, G, cutoffs=(1, 5, 25, 100)):
    results = {c: 0 for c in cutoffs}

    for row_I, y in zip(I, G):

        for c in cutoffs:
            p = set(row_I[:c])
            results[c] += float(len(p & y) > 0)

    for k, v in results.items():
        results[k] = v / len(G)

    return results