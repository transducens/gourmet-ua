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
    prev_output_factors =None
    cur_output_factors =None
    target = None

    asyncTLFactors=False
    if samples[0].get('target', None) is not None:

        if samples[0].get('target_factors_async', None) is not None:
            asyncTLFactors=True

        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)

        target_factors=merge('target_factors', left_pad=left_pad_target)
        target_factors = target_factors.index_select(0, sort_order)

        if asyncTLFactors:
            target_factors_async=merge('target_factors_async', left_pad=left_pad_target)
            target_factors_async = target_factors_async.index_select(0, sort_order)

        ntokens = sum(len(s['target']) for s in samples)+sum(len(s['target_factors']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

            cur_output_factors= merge(
                'target_factors',
                left_pad=left_pad_target,
                move_eos_to_beginning=False,
            )
            cur_output_factors = cur_output_factors.index_select(0, sort_order)

            if asyncTLFactors:
                prev_output_factors=merge(
                    'target_factors_async',
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                )
            else:
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
    if asyncTLFactors:
        batch['target_factors_async'] = target_factors_async
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    if prev_output_factors is not None:
        batch['net_input']['prev_output_factors'] = prev_output_factors
    if cur_output_factors is not None:
        batch['net_input']['cur_output_factors'] = cur_output_factors
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
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,tgt_factors_async=None,tgt_factors_async_sizes=None
    ):
        super().__init__(src, src_sizes, src_dict,
        tgt, tgt_sizes, tgt_dict,
        left_pad_source,left_pad_target,
        max_source_positions, max_target_positions,
        shuffle, input_feeding, remove_eos_from_source, append_eos_to_target)

        self.tgt_factors=tgt_factors
        self.tgt_factors_sizes=tgt_factors_sizes
        self.tgt_factors_dict=tgt_factors_dict

        self.tgt_factors_async=tgt_factors_async
        self.tgt_factors_async_sizes=tgt_factors_async_sizes

    def __getitem__(self, index):
        d=super().__getitem__(index)
        tgt_factors_item =self.tgt_factors[index] if self.tgt_factors is not None else None
        tgt_factors_async_item=self.tgt_factors_async[index] if self.tgt_factors_async is not None else None
        if self.append_eos_to_target:
            eos = self.tgt_factors_dict.eos() if self.tgt_factors_dict else self.src_dict.eos()
            if self.tgt_factors and self.tgt_factors[index][-1] != eos:
                tgt_factors_item = torch.cat([self.tgt_factors[index], torch.LongTensor([eos])])
            if self.tgt_factors_async and self.tgt_factors_async[index][-1] != eos:
                tgt_factors_async_item = torch.cat([self.tgt_factors_async[index], torch.LongTensor([eos])])

        d['target_factors']=tgt_factors_item
        d['target_factors_async']=tgt_factors_async_item
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
