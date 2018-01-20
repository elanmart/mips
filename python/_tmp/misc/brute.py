def read_file(name):
    lines = open(name).readlines()
    cnt, dim = [int(s) for s in lines[0].split()]

    data = []
    for i in range(cnt):
        data.append([float(s) for s in lines[i+1].split()])

    return data

data = read_file("input")
queries = read_file("queries")

for qnum, q in enumerate(queries):
    values = []
    
    for vnum, v in enumerate(data):
        s = sum(a * b for a, b in zip(q, v))
        values.append( (s, vnum) )

    print("Best for query".format(qnum))

    for val in sorted(values)[::-1]:
        print('{}\t{}'.format(val[1], val[0]))

    print()
