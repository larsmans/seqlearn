from __future__ import absolute_import
import numpy as np

class SequenceShards(object):
    """Sequence-aware (repeated) splitter.

    Uses a greedy heuristic to partition input sequences into sets with roughly
    equal numbers of samples, while keeping the sequences intact.

    Parameters
    ----------
    lengths : array-like of integers, shape (n_samples,)
        Lengths of sequences, in the order in which they appear in the dataset.

    n_folds : int, optional
        Number of folds.

    Returns
    -------
    A generator yielding (indices, length_indices) tuples.
    """

    def __init__(self, lengths, n_folds):
        self.lengths = lengths
        self.n_folds = n_folds

    def __iter__(self):
        lengths = np.asarray(self.lengths, dtype=np.intp)
        starts = np.cumsum(lengths) - lengths
        n_samples = np.sum(lengths)

        seq_ind = np.arange(len(lengths))

        folds = [[] for _ in range(self.n_folds)]
        samples_per_fold = np.zeros(self.n_folds, dtype=int)

        # Greedy strategy: always append to the currently smallest fold
        for i in seq_ind:
            seq = (i, starts[i], starts[i] + lengths[i])
            fold_idx = np.argmin(samples_per_fold)
            folds[fold_idx].append(seq)
            samples_per_fold[fold_idx] += lengths[i]

        for f in folds:
            mask = np.zeros(n_samples, dtype=bool)
            lengths_mask = np.zeros(len(lengths), dtype=bool)
            for i, start, end in f:
                mask[start:end] = True
                lengths_mask[i] = True
            indices = np.where(mask)[0]
            length_indices = np.where(lengths_mask)[0]
            yield indices, length_indices

    def __len__(self):
        return self.n_folds
