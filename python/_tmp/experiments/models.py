import logging

import torch as th

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def noop(x):
    return x


class FeedForward(th.nn.Module):
    def __init__(self, num_embeddings, num_labels,
                 emb_dim=256, hid_dim=512,
                 emb_activation=noop, hid_activation=th.nn.SELU,
                 init_range=0.05):

        super().__init__()
        self.init_range = init_range

        self.embedding = th.nn.EmbeddingBag(num_embeddings, emb_dim)
        self.activation = emb_activation
        self.decoder = th.nn.Sequential(

            hid_activation(),
            th.nn.BatchNorm1d(emb_dim),

            th.nn.Linear(emb_dim, hid_dim),
            hid_activation(),
            th.nn.BatchNorm1d(hid_dim),

            th.nn.Linear(hid_dim, hid_dim),
            hid_activation(),
            th.nn.BatchNorm1d(hid_dim),

            th.nn.Linear(hid_dim, num_labels)
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.weight.data.uniform_(-self.init_range, self.init_range)

    def forward(self, indices, offsets):
        E = self.embedding(indices, offsets)
        A = self.activation(E)
        O = self.decoder(A)

        return O
