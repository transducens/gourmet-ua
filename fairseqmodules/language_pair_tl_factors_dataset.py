import fairseq.data

from fairseq.data import data_utils, FairseqDataset


#import utils
from fairseq import utils


import numpy as np
import torch

def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        target_factors=merge('target_factors', left_pad=left_pad_target)
        target_factors = target_factors.index_select(0, sort_order)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

            prev_output_factors = merge(
                'target_factors',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_factors = prev_output_factors.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'target_factors': target_factors
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    if prev_output_factors is not None:
        batch['net_input']['prev_output_factors'] = prev_output_factors
    return batch

class LanguagePairTLFactorsDataset(fairseq.data.LanguagePairDataset):
    """
        Methods not overridden:
        num_tokens: returns max tokens among src and tgt
        size: returns tuple (src_size,tgt_size), used to filter with --max-positions
        ordered_indices: sort indexes by length

    """
    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        tgt_factors=None,tgt_factors_sizes=None, tgt_factors_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
    ):
        super().__init__(src, src_sizes, src_dict,
        tgt, tgt_sizes, tgt_dict,
        left_pad_source,left_pad_target,
        max_source_positions, max_target_positions,
        shuffle, input_feeding, remove_eos_from_source, append_eos_to_target)

        self.tgt_factors=tgt_factors
        self.tgt_factors_sizes=tgt_factors_sizes
        self.tgt_factors_dict=tgt_factors_dict


    def __getitem__(self, index):
        d=super().__getitem__(index)
        tgt_factors_item =self.tgt_factors[index] if self.tgt_factors is not None else None
        if self.append_eos_to_target:
            eos = self.tgt_factors_dict.eos() if self.tgt_factors_dict else self.src_dict.eos()
            if self.tgt_factors and self.tgt_factors[index][-1] != eos:
                tgt_factors_item = torch.cat([self.tgt_factors[index], torch.LongTensor([eos])])

        d['target_factors']=tgt_factors_item
        return d

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch with the following keys:
                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:
                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `prev_output_factors` (LongTensor): a padded 2D Tensor of
                      factors in the target sentence, shifted right by one position
                      for input feeding/teacher forcing, of shape `(bsz,
                      tgt_factors_len)`. This key will not be present if *input_feeding*
                      is ``False``. Padding will appear on the left if
                      *left_pad_target* is ``True``.
                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `target_factors` (LongTensor): a padded 2D Tensor of factors in the
                  target sentence of shape `(bsz, tgt_factors_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        bsz = max(num_tokens // max(src_len, tgt_len), 1)
        return self.collater([
            {
                'id': i,
                'source': self.src_dict.dummy_sentence(src_len),
                'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
                'target_factors':self.tgt_factors_dict.dummy_sentence(tgt_len) if self.tgt_factors_dict is not None else None
            }
            for i in range(bsz)
        ])

    @property
    def supports_prefetch(self):
        return super().supports_prefetch and getattr(self.tgt_factors, 'supports_prefetch', False)

    def prefetch(self, indices):
        super().prefetch()
        self.tgt_factors.prefetch(indices)
