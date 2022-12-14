# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch.utils.data

from fairseq.data import data_utils


class EpochListening:
    """Mixin for receiving updates whenever the epoch increments."""

    def set_epoch(self, epoch):
        """Will receive the updated epoch number at the beginning of the epoch."""
        pass


class FairseqDataset(torch.utils.data.Dataset, EpochListening):
    """A dataset that provides helpers for batching."""

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        raise NotImplementedError

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def src_num_tokens(self,index):
        raise NotImplementedError

    def tgt_num_tokens(self,index):
        raise NotImplementedError

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        raise NotImplementedError

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self), dtype=np.int64)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False

    def attr(self, attr: str, index: int):
        return getattr(self, attr, None)

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        raise NotImplementedError

    def get_batch_shapes(self):
        """
        Return a list of valid batch shapes, for example::

            [(8, 512), (16, 256), (32, 128)]

        The first dimension of each tuple is the batch size and can be ``None``
        to automatically infer the max batch size based on ``--max-tokens``.
        The second dimension of each tuple is the max supported length as given
        by :func:`fairseq.data.FairseqDataset.num_tokens`.

        This will be used by :func:`fairseq.data.FairseqDataset.batch_by_size`
        to restrict batch shapes. This is useful on TPUs to avoid too many
        dynamic shapes (and recompilations).
        """
        return None

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
        my_batching=0,
        tol=0
    ):
        """
        Given an ordered set of indices, return batches according to
        *max_tokens*, *max_sentences* and *required_batch_size_multiple*.
        """
        from fairseq.data import data_utils

        fixed_shapes = self.get_batch_shapes()
        if fixed_shapes is not None:

            def adjust_bsz(bsz, num_tokens):
                if bsz is None:
                    assert max_tokens is not None, 'Must specify --max-tokens'
                    bsz = max_tokens // num_tokens
                if max_sentences is not None:
                    bsz = min(bsz, max_sentences)
                elif (
                    bsz >= required_batch_size_multiple
                    and bsz % required_batch_size_multiple != 0
                ):
                    bsz -= (bsz % required_batch_size_multiple)
                return bsz

            fixed_shapes = np.array([
                [adjust_bsz(bsz, num_tokens), num_tokens]
                for (bsz, num_tokens) in fixed_shapes
            ])
        #print(1, len(indices),type(indices))#4500966
        #a = [self.num_tokens(i)>=20 and self.num_tokens(i)<=40 for i in range(len(indices))]
        #print(np.sum(np.array(a))) #2248033
        '''
        new_indices = []
        for i in range(len(indices)):
            if (self.src_num_tokens(indices[i])>=25 and self.src_num_tokens(indices[i])<=35):
                new_indices.append(indices[i])
        indices = np.array(new_indices)
        '''
        #print(2, len(indices))
        #exit()
        #max(src,tgt):
        #20-40:2248033
        #25-35:1209236
        #27-32:673342
        #src:
        #20-40:2243518
        #25-35:1188508
        #27-32:663084

        return data_utils.batch_by_size(
            indices,
            num_tokens_fn=self.num_tokens,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            fixed_shapes=fixed_shapes,
            my_batching=my_batching,
            tol=tol,
        )

    def filter_indices_by_size(self, indices, max_sizes):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        WARNING: don't update, override method in child classes

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, 'sizes') and isinstance(self.sizes, np.ndarray):
                ignored = indices[self.sizes[indices] > max_sizes].tolist()
                indices = indices[self.sizes[indices] <= max_sizes]
            elif hasattr(self, 'sizes') and isinstance(self.sizes, list) and len(self.sizes) == 1:
                ignored = indices[self.sizes[0][indices] > max_sizes].tolist()
                indices = indices[self.sizes[0][indices] <= max_sizes]
            else:
                indices, ignored = data_utils._filter_by_size_dynamic(indices, self.size, max_sizes)
        else:
            indices, ignored = data_utils._filter_by_size_dynamic(indices, self.size, max_sizes)
        return indices, ignored


class FairseqIterableDataset(torch.utils.data.IterableDataset, EpochListening):
    """
    For datasets that need to be read sequentially, usually because the data is
    being streamed or otherwise can't be manipulated on a single machine.
    """

    def __iter__(self):
        raise NotImplementedError
