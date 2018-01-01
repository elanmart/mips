#!/usr/bin/env python3

import csv
import ast
import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple

with open("results/df-30-12-2017_24_00.csv") as f:
    s = f.read()
    nl = s.find("\n")
    with open("results/no_first_line.csv", "w") as w:
        w.write(s[nl+1:])

results = defaultdict(dict)

parameter_tuples = {
    "Flat": namedtuple("FlatParameters", []),
    "IVF": namedtuple("IVFParameters", ["nprobe", "size"]),
    "KMeans": namedtuple("KMeansParameters", ["U", "bnb",
        "layers", "m", "nprobe", "nprobe_test", "spherical"]),
    "Alsh": namedtuple("IVFParameters", ["K", "L", "U", "m", "r"]),
    "Quant": namedtuple("IVFParameters", [
        "centroid_count", "subspace_count"]),
}

def add(results, row):
    if row["status"] == "FAIL":
        return

    if row["timeout"] == "True":
        return

    index = row["cls_name"].replace("Index", "")
    data = row["data_path"].rstrip("/").split("/")[-1]
    key = (index, data)

    tup = parameter_tuples[key[0]]
    params = tup(**{k: row[k] for k in tup._fields})

    try:
        value = {
            "p100" : float(row["p_at_00100"]),
            "p25"  : float(row["p_at_00025"]),
            "p5"   : float(row["p_at_00005"]),
            "p1"   : float(row["p_at_00001"]),
            "train": float(row["train-time"]),
            "test" : float(row["test-time"]),
        }
    except ValueError:
        return
    results[key][params] = value

with open("results/no_first_line.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        if row["cls_name"] == "KMeansIndex":
            tests = ast.literal_eval(row["nprobe_test"])
            for t in tests:
                row2 = row.copy()
                row2["nprobe_test"] = t
                for k in row2:
                    if k.endswith("-%05d" % t):
                        k2 = k[:k.rfind("-")]
                        row2[k2] = row2[k]
                add(results, row2)
        else:
            add(results, row)

indexes = set(k[0] for k in results)
datasets = set(k[1] for k in results)
precisions = [1, 5, 25, 100]

def make_nondominated():
    for p in precisions:
        pd = "p%d" % p
        for d in datasets:
            graph_data = defaultdict(list)
            for i in indexes:
                res = results[(i, d)]
                for row in res.values():
                    dominated = False
                    for other in res.values():
                        if other["test"] < row["test"] and other[pd] > row[pd]:
                            dominated = True
                    if not dominated:
                        graph_data[i].append((row["test"], row[pd]))
                graph_data[i].sort()
                print(i, d, len(graph_data[i]), len(res))
            
            plt.clf()
            lines = []
            for i in graph_data:
                t = [k[0] for k in graph_data[i]]
                prec = [k[1] for k in graph_data[i]]
                lines.append(plt.plot(t, prec, label = i, marker = "o")[0])
            plt.legend(handles = lines)
            plt.savefig("graphs/nondominated-%s-%d.png" % (d, p))

make_nondominated()
