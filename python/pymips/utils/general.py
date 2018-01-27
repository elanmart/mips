import time
from contextlib import contextmanager


class Clock:
    def __init__(self):
        self.start    = time.time()
        self._elapsed = None

    @property
    def elapsed(self):

        if self._elapsed is None:
            self.stop()

        return self._elapsed

    def stop(self):
        self._elapsed = time.time() - self.start


@contextmanager
def timer():
    c = Clock()
    yield c
    c.stop()
