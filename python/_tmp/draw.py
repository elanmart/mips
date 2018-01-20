import argparse
import csv

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot

MARKER_SIZE = 3.5
WIDTH     = 0.5
TIME_DS   = 8
RECALL_DS = 6  # recall at 10
INTER_DS  = 9  # intersection
N_QUERIES = 100
ALGORITHM_LIST = ['ivf', 'kmeans', 'quant', 'alsh']

def create_plot(plot_label, plot_type, fig, host, data, limits):
    labels = [plot_label + ' - ' + str(x) for x in ALGORITHM_LIST]
    if plot_type == 'recall':
        markers = ['ro', 'yo', 'bo', 'go']
        DS = RECALL_DS
    elif plot_type == 'inter':
        markers = ['rs', 'ys', 'bs', 'gs']
        DS = INTER_DS

    plt.suptitle(str(plot_label) + ' vs. time')
    host.set_ylabel(str(plot_label))
    host.set_ylim(0.9 * limits[0], 1.1 * limits[1])

    with open(str(plot_type) + '_plot.html', 'w') as html:
        html.write('<!DOCTYPE html><html><head>\n')
        html.write('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>')
        html.write('</head><body><div id="myDiv"></div><script>')

        for i, alg in enumerate(ALGORITHM_LIST):
            html.write('var data' + str(i) + ' = {\n x: ')
            x = [float(d[TIME_DS]) * 1000 / N_QUERIES for d in data[i]]
            y = [d[DS] for d in data[i]]
            html.write(str(x) + ',\n y: ' + str([float(a) for a in y]))
            if alg == 'ivf':
                text = ['nprobe=' + str(d[0]) for d in data[i]]
            elif alg == 'kmeans':
                text = ['layers=' + str(d[0]) + ' op_tr=' + str(d[4]) +
                        ' aug_type=' + str(d[1]) + ' U=' + str(d[3])
                        for d in data[i]]
            elif alg == 'quant':
                text = ['subsp=' + str(d[0]) + ' centr=' + str(d[1])
                        for d in data[i]]
            elif alg == 'alsh':
                text = ['tabl=' + str(d[0]) + ' fun=' + str(d[1]) +
                        ' aug_type=' + str(d[3]) + ' U=' + str(d[4]) +
                        ' r=' + str(d[2]) for d in data[i]]
            html.write(',\n text: ' + str(text) + ',\n mode: \'markers\',\n')
            html.write('name :\'' + str(alg) + '\' };\n')

            host.plot(x, y, markers[i], label=labels[i],
                      markersize=MARKER_SIZE, mfc='none', markeredgewidth=WIDTH)

        html.write('\nvar data = [data0, data1, data2, data3];\n\nvar layout = { title:\'')
        html.write(str(plot_label) + ' vs. time\', yaxis: { title: \'' + str(plot_label))
        html.write('\' }, xaxis: { type: \'log\', autorange: true, title: \'Time per query [ms]\' }, height : 1000 };\n\n')
        html.write('Plotly.newPlot(\'myDiv\', data, layout);\n</script></body></html>')

    host.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.savefig(str(plot_type) + '_plot.pdf')

def main(args):

    with open(args.input) as f:
        reader = csv.reader(f, delimiter="\t")
        inlist = list(reader)

    minimum_time   = min([float(d[TIME_DS + 1]) for d in inlist])
    maximum_time   = max([float(d[TIME_DS + 1]) for d in inlist])
    minimum_recall = min([float(d[RECALL_DS + 1]) for d in inlist])
    maximum_recall = max([float(d[RECALL_DS + 1]) for d in inlist])
    minimum_inter  = min([float(d[INTER_DS + 1]) for d in inlist])
    maximum_inter  = max([float(d[INTER_DS + 1]) for d in inlist])

    data = []
    for algorithm in ALGORITHM_LIST:
        data.append([x[1:] for x in inlist if x[0] == algorithm])

    # prepare the plot
    fig = plt.figure()
    host = host_subplot(111)
    box = host.get_position()
    host.set_position([box.x0, box.y0 + box.height*0.35, box.width, box.height * 0.7])
    host.set_xscale('log')
    host.set_xlabel("Time per query [ms]")
    host.set_xlim(0.9 * 1000 * minimum_time / N_QUERIES,
                  1.1 * 1000 * maximum_time / N_QUERIES)

    if args.mode == 'recall':
        create_plot('Recall@10', 'recall', fig=fig, host=host, data=data, limits=(minimum_recall, maximum_recall))

    if args.mode == 'inter':
        create_plot('Top-100 intersection', 'inter', fig=fig, host=host, data=data, limits=(minimum_inter, maximum_inter))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("A simple utility to create plots with benchmark results.")
    parser.add_argument("--input", default="output.txt",
                        help="File from which the data should be read.")
    parser.add_argument("--mode", choices={"recall", "inter"}, default="recall",
                        help="Controls what plot will be drawn")
    args = parser.parse_args()

    main(args)
