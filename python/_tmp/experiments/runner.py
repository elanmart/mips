from experiments.models import noop
from experiments.utils import path_finder
from .training import main, DefaultOptions
import torch as th
from torch.nn import functional as F


class Options(DefaultOptions):
    NAME          = "test"
    LOGGING_ROOT  = path_finder('../../data/lshtc-logs')
    PATH          = path_finder('../../data/LSHTC')

    N_EPOCHS      = 3

    WEIGHTS       = False
    MODE          = None

    LR            = 1e-3

    CUDA          = True
    SAMPLE_Y      = False

    LOSS          = F.binary_cross_entropy_with_logits
    # LOSS          = F.cross_entropy


lrs = [1e-3, 1e-3, 1e-5]

ls = [
    (F.cross_entropy, True),
    (F.binary_cross_entropy_with_logits, False),
]

wms = [
    (False, None),
    (True, None),
    (True, 'full'),
]
eas = [noop, F.tanh]
has = [th.nn.SELU]


SKIP = 0

if __name__ == '__main__':
    skipped = 0

    for lr in lrs:
        for w, m in wms:
            for ea in eas:
                for ha in has:
                    for loss_fn, sample in ls:

                        if skipped < SKIP:
                            skipped += 1
                            continue

                        Options.NAME = f"|sm-{sample}|loss-{loss_fn.__name__}" \
                                       f"|lr-{lr}|w-{w}|m-{m}|ea-{ea.__name__}|ha-{ha.__name__}|"
                        Options.LOSS = loss_fn
                        Options.SAMPLE_Y = sample
                        Options.LR = lr
                        Options.WEIGHTS = w
                        Options.MODE = m
                        Options.EMB_ACTIV = ea
                        Options.HID_ACTIV = ha

                        main(Options)
