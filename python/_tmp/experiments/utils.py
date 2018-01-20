import logging
import os

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def path_finder(path):
    file_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(file_path, path))


def run_ID(filename='/tmp/__TORCH_RUN_IDX'):
    _IDX = 1

    if os.path.exists(filename):
        _IDX = int(open(filename).read().strip()) + 1

    with open(filename, 'w') as f:
        f.write(str(_IDX))

    logging.info(f"Fetched run index: {_IDX}")
    return f' [run_{_IDX}]'