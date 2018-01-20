import logging
import os

import numpy as np
import torch as th
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from experiments.data import get_data, to_cuda_var, get_weights, CSRDataset, Preprocessor
from experiments.models import FeedForward, noop
from experiments.utils import path_finder, run_ID

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class DefaultOptions:

    @classmethod
    def to_string(cls):
        for item in sorted(dir(cls)):

            if not item.isupper():
                continue

            yield f"`{item}:\t{str(getattr(cls, item))}`"

    NAME          = "DefaultConfig"

    SEED          = 42
    CUDA          = True

    LOGGING_ROOT  = path_finder('../../data/logs')
    PATH          = path_finder('../../data/Wiki10')

    BATCH_SIZE    = 64
    N_EPOCHS      = 1

    EMB_DIM       = 256
    HID_DIM       = 512
    DROPOUT       = 0.

    EMB_ACTIV     = noop
    HID_ACTIV     = th.nn.SELU
    INIT_RANGE    = 0.05

    OPTIMIZER     = th.optim.Adam
    LR            = 1e-3
    LOSS          = F.binary_cross_entropy_with_logits

    SCALE_Y       = False
    SAMPLE_Y      = False

    WEIGHTS       = False
    MODE          = None  # {'full', 'row', 'sqrt', 'sqrt-pos')

    NUM_WORKERS   = 0 if (not CUDA) else os.cpu_count() - 1
    PIN_MEMORY    = True

    PRINT_ACC     = 10
    PRINT_HIST    = 100

    SAMPLER       = RandomSampler
    DEBUG_SHUF    = False

    EVAL          = True


def train(loader, test_loader,
          model, loss_fn, optimizer,
          weights,
          Config: DefaultOptions):

    writer = SummaryWriter(os.path.join(Config.LOGGING_ROOT, Config.NAME + run_ID()))
    logging.info("Initiating training loop...")
    for idx, param in enumerate(Config.to_string()):
        writer.add_text('params', param, global_step=idx)

    try:
        for epoch in range(Config.N_EPOCHS):
            model.train()

            logging.info(f"Epoch: {epoch}")
            running_loss = 0.

            for i, data in enumerate(loader):

                (indices, offsets), y = data
                optimizer.zero_grad()

                if Config.DEBUG_SHUF:
                    indices = indices[th.randperm(len(indices)).long()]

                indices, offsets, y = to_cuda_var(indices, offsets, y, cuda=Config.CUDA)
                outputs = model(indices, offsets)

                if weights is not None:
                    pos_weights, neg_weights = weights
                    pos_weights, neg_weights = pos_weights.detach(), neg_weights.detach()
                    if Config.SAMPLE_Y:
                        W = pos_weights.squeeze()
                    else:
                        W = y * pos_weights + (1 - y) * neg_weights
                else:
                    W = None

                loss = loss_fn(outputs, y, weight=W)

                loss.backward()
                optimizer.step()

                # = = = = = = = = = = #
                # = = = LOGGING = = = #
                running_loss += loss.data[0]
                global_step   = epoch * len(loader) + i

                writer.add_scalar(f'metrics/loss', loss.data[0], global_step=global_step)

                if Config.PRINT_ACC > 0 and (i % Config.PRINT_ACC) == 0:

                    __labels  = y.data.cpu().numpy()
                    __outputs = outputs.data.cpu().numpy()

                    __best    = np.argmax(__outputs, 1)
                    __relevant = (__labels == __best) if Config.SAMPLE_Y else __labels[np.arange(len(__best)), __best]
                    __score    = np.mean(__relevant)

                    # noinspection PyTypeChecker
                    writer.add_scalar(f'metrics/p_at_1', float(__score), global_step=global_step)

                    # for name, p in model.named_parameters():
                    #     writer.add_histogram('weights/'+name, p.data.cpu(), global_step)
                    #     writer.add_histogram('grads/'+name, p.grad.data.cpu(), global_step)

                del indices, offsets, outputs, loss

            if Config.EVAL:
                vscore = evaluate(test_loader, model, Config)
                writer.add_scalar(f'valid_metrics/p_at_1', float(vscore), global_step=epoch)

            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.

        th.save(model, 'model.th')

    except KeyboardInterrupt:
        th.save(model, 'model.th')
        raise

    logging.info("bye.")


def evaluate(loader, model,
             Config: DefaultOptions):

    model.eval()
    p_at_1 = 0.

    for i, data in enumerate(loader):

        (indices, offsets), y = data

        indices, offsets, y = to_cuda_var(indices, offsets, y, cuda=Config.CUDA, volatile=True)
        outputs = model(indices, offsets)

        __labels = y.data.cpu().numpy()
        __outputs = outputs.data.cpu().numpy()

        __best = np.argmax(__outputs, 1)
        __relevant = (__labels == __best) if Config.SAMPLE_Y else __labels[np.arange(len(__best)), __best]
        __score = np.sum(__relevant)

        p_at_1 += __score

    p_at_1 /= (len(loader) * Config.BATCH_SIZE)

    return p_at_1


def main(Options):
    th.manual_seed(Options.SEED)
    th.cuda.manual_seed(Options.SEED)

    X, Y   = get_data(Options.PATH, 'train')
    Xt, Yt = get_data(Options.PATH, 'test')
    weights = None

    if Options.WEIGHTS:
        weights = to_cuda_var(get_weights(Y, mode=Options.MODE), cuda=Options.CUDA, var=True)

    dataset = CSRDataset(X, Y)
    sampler = Options.SAMPLER(dataset)
    preproc = Preprocessor(n_labels=Y.shape[1], scale_y=Options.SCALE_Y, sample_y=Options.SAMPLE_Y)
    loader  = DataLoader(dataset=dataset, batch_size=Options.BATCH_SIZE, sampler=sampler, collate_fn=preproc.collate_fn,
                         num_workers=Options.NUM_WORKERS, pin_memory=Options.PIN_MEMORY)

    test_dataset = CSRDataset(Xt, Yt)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=Options.BATCH_SIZE, sampler=SequentialSampler(test_dataset),
                              collate_fn=preproc.collate_fn, num_workers=Options.NUM_WORKERS, pin_memory=Options.PIN_MEMORY)

    model = to_cuda_var(
        FeedForward(num_embeddings=X.shape[1]+1, num_labels=Y.shape[1],
                    emb_dim=Options.EMB_DIM, hid_dim=Options.HID_DIM,
                    emb_activation=Options.EMB_ACTIV, hid_activation=Options.HID_ACTIV,
                    init_range=Options.INIT_RANGE),
        cuda=Options.CUDA, var=False
    )
    loss_fn = Options.LOSS
    optimizer = Options.OPTIMIZER(model.parameters(), lr=Options.LR)

    train(loader=loader, test_loader=test_loader,
          model=model, loss_fn=loss_fn, optimizer=optimizer,
          weights=weights,
          Config=Options)


if __name__ == '__main__':
    main(DefaultOptions)