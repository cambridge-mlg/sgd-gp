from annoy import AnnoyIndex
import random
import os
import jax.numpy as jnp
import numpy as np


def create_annoy(x, n_trees=10, savefile=None, recompute=False):
    assert len(x.shape) == 2

    d = x.shape[-1]  # Length of item vector that will be indexed

    t = AnnoyIndex(d, "euclidean")
    if savefile is None or not os.path.isfile(savefile) or recompute:
        print("created index")
        for i, xi in enumerate(x):
            t.add_item(i, xi.tolist())
        print("loaded data")
        t.build(n_trees)  #
        print("built model")
        if savefile is not None:
            t.save(savefile)
            print("saved model")
    else:
        t.load(savefile)
        print("loaded model")

    return t


def generate_close_pair_dict(data, t, num_points, num_neighbours, max_dist):
    close_pairs = {}

    for i in range(num_points):
        NNs = t.get_nns_by_item(i, num_neighbours)
        if i in NNs:
            NNs.remove(i)
        NNs = np.array(NNs)
        dists = ((data[i][None, :] - data[NNs]) ** 2).sum(
            axis=-1
        ) ** 0.5  # (num_neighbours,)
        entry_indices = np.where(dists < max_dist)[0]
        close_pairs[i] = set(NNs[entry_indices])  # .tolist()

    return close_pairs


def get_pruned_indices(close_pairs):
    kept = set()
    deleted = set()

    keys = list(close_pairs.keys())

    for i in keys:
        if len(close_pairs[i]) > 1:
            if i not in kept:
                deleted.add(i)

            for n in close_pairs[i]:
                if n not in deleted:
                    kept.add(n)
                    ball = close_pairs[i].intersection(close_pairs[n])
                    deleted.update(ball.difference(kept))

        else:
            kept.add(i)

    return list(kept.difference(deleted))


def annoy_cluster_dataset(
    data, n_trees=15, num_neighbours=15, max_dist=2, savefile=None, recompute=False
):
    model = create_annoy(data.x, n_trees, savefile, recompute)
    num_points = data.x.shape[0]
    close_pairs = generate_close_pair_dict(
        data.x, model, num_points, num_neighbours, max_dist
    )
    print("created pair dict")
    pruned_indices = get_pruned_indices(close_pairs)
    print(f"obtained {len(pruned_indices)} non-pruned indices")
    data.z = data.x[jnp.array(pruned_indices)]
    return data, pruned_indices
