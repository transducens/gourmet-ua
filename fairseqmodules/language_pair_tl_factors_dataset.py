import fairseq.data

from fairseq.data import data_utils, FairseqDataset


#import utils
from fairseq import utils


import numpy as np
import torch

WAIT="<<WAIT>>"

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

    src_factors=None
    src_factors_lengths=None

    #Source factors
    if samples[0].get('source_factors_async', None) is not None:
        src_factors = merge('source_factors_async', left_pad=left_pad_source)
        src_factors = src_factors.index_select(0, sort_order)

        src_factors_lengths=torch.LongTensor([s['source_factors_async'].numel() for s in samples])
        src_factors_lengths=src_factors_lengths.index_select(0, sort_order)

    prev_output_tokens = None
    prev_output_tokens_lengths = None
    prev_output_factors =None
    cur_output_factors =None
    target = None
    target_factors=None
    prev_output_tokens_first_subword=None
    prev_output_tokens_last_subword=None
    prev_output_tokens_word_end_positions=None

    asyncTLFactors=False
    waitActionReplace=False
    if samples[0].get('target_factors_no_wait', None) is not None:
        waitActionReplace=True
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

        ntokens_a=sum(len(s['target']) for s in samples)
        ntokens_b=sum(len(s['target_factors']) for s in samples)
        ntokens =ntokens_a+ntokens_b


        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

            key_to_merge= 'target_factors_no_wait' if waitActionReplace else  'target_factors'
            cur_output_factors= merge(
                key_to_merge,
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

                prev_output_tokens_lengths=torch.LongTensor([s['target'].numel() for s in samples])
                prev_output_tokens_lengths=prev_output_tokens_lengths.index_select(0, sort_order)

                #fill field position_target_word_ends
                #1. sort according to sort_order
                prev_output_tokens_word_end_positions_unsorted=[ s['position_target_word_ends'] for s in samples ]
                prev_output_tokens_word_end_positions=[ prev_output_tokens_word_end_positions_unsorted[i] for i in sort_order ]

                #2. shift one position because prev_output_tokens were shifted too
                for i in range(len(prev_output_tokens_word_end_positions)):
                    prev_output_tokens_word_end_positions[i]=[ 0 ] + [ p+1 for p in  prev_output_tokens_word_end_positions[i]]

            else:
                prev_output_factors = merge(
                    'target_factors',
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                )

            prev_output_factors = prev_output_factors.index_select(0, sort_order)

            #
            if samples[0].get('target_only_first_subword', None) is not None:
                prev_output_tokens_first_subword= merge(
                    'target_only_first_subword',
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                )
                prev_output_tokens_first_subword = prev_output_tokens_first_subword.index_select(0, sort_order)

            if samples[0].get('target_only_last_subword', None) is not None:
                prev_output_tokens_last_subword= merge(
                    'target_only_last_subword',
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                )
                prev_output_tokens_last_subword = prev_output_tokens_last_subword.index_select(0, sort_order)


    else:
        ntokens = sum(len(s['source']) for s in samples)
        ntokens_a=ntokens
        ntokens_b=ntokens

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'ntokens_a': ntokens_a,
        'ntokens_b': ntokens_b,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'target_factors': target_factors
    }
    if asyncTLFactors:
        batch['target_factors_async'] = target_factors_async
    if src_factors is not None:
        batch['net_input']['src_factors'] = src_factors
    if src_factors_lengths is not None:
        batch['net_input']['src_factors_lengths'] = src_factors_lengths
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    if prev_output_factors is not None:
        batch['net_input']['prev_output_factors'] = prev_output_factors
    if cur_output_factors is not None:
        batch['net_input']['cur_output_factors'] = cur_output_factors
    if prev_output_tokens_first_subword is not None:
        batch['net_input']['prev_output_tokens_first_subword'] = prev_output_tokens_first_subword
    if prev_output_tokens_last_subword is not None:
        batch['net_input']['prev_output_tokens_last_subword'] = prev_output_tokens_last_subword
    if prev_output_tokens_lengths is not None:
        batch['net_input']['prev_output_tokens_lengths'] = prev_output_tokens_lengths
    if prev_output_tokens_word_end_positions is not None:
        batch['net_input']['prev_output_tokens_word_end_positions'] = prev_output_tokens_word_end_positions
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
        tgt_factors_async=None,tgt_factors_async_sizes=None,
        src_factors_async=None,src_factors_async_sizes=None,src_factors_dict=None,
        tgt_only_first_subword=None,tgt_only_first_subword_sizes=None,
        tgt_only_last_subword=None,tgt_only_last_subword_sizes=None,
        add_wait_action=False,replace_wait_at_sf_input=False
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

        self.src_factors_dict=src_factors_dict
        self.src_factors_async=src_factors_async
        self.src_factors_async_sizes=src_factors_async_sizes

        self.tgt_only_first_subword=tgt_only_first_subword
        self.tgt_only_first_subword_sizes=tgt_only_first_subword_sizes
        self.tgt_only_last_subword=tgt_only_last_subword
        self.tgt_only_last_subword_sizes=tgt_only_last_subword_sizes

        self.add_wait_action=add_wait_action
        self.replace_wait_at_sf_input=replace_wait_at_sf_input

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

        src_factors_async_item=self.src_factors_async[index] if self.src_factors_async is not None else None
        tgt_only_first_subword_item=self.tgt_only_first_subword[index] if self.tgt_only_first_subword is not None else None
        tgt_only_last_subword_item=self.tgt_only_last_subword[index] if self.tgt_only_last_subword is not None else None

        d['target_factors']=tgt_factors_item
        if tgt_factors_async_item is not None:
            d['target_factors_async']=tgt_factors_async_item

        if src_factors_async_item is not None:
            d['source_factors_async']=src_factors_async_item

        if tgt_only_first_subword_item is not None:
            d['target_only_first_subword']=tgt_only_first_subword_item
        if tgt_only_last_subword_item is not None:
            d['target_only_last_subword']=tgt_only_last_subword_item

        d['position_target_word_ends']=[ i for i,w in enumerate(d['target']) if len(self.tgt_dict.string([w]).strip()) > 0 and  not self.tgt_dict.string([w]).endswith("@@") ]

        if self.add_wait_action and self.replace_wait_at_sf_input :
            #create a new target_factors in which the WAIT actions are replaced by the corresponding token
            d['target_factors_no_wait']=tgt_factors_item.clone().detach()
            prevToken=None
            for i in range(len(d['target_factors_no_wait'])):
                if d['target_factors_no_wait'][i] != self.tgt_factors_dict.index(WAIT):
                    prevToken=d['target_factors_no_wait'][i]
                else:
                    d['target_factors_no_wait'][i]=prevToken

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
                'target_factors':self.tgt_factors_dict.dummy_sentence(tgt_len) if self.tgt_factors_dict is not None else None,
                'target_factors_async':self.tgt_factors_dict.dummy_sentence(tgt_len) if self.tgt_factors_async and self.tgt_factors_dict is not None else None,
                'source_factors_async':self.src_factors_dict.dummy_sentence(src_len) if self.src_factors_async and self.src_factors_dict is not None else None,
                'target_only_first_subword': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_only_first_subword and self.tgt_dict is not None else None,
                'target_only_last_subword': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_only_last_subword and self.tgt_dict is not None else None,
                'position_target_word_ends': [ i for i in range(tgt_len-1) ]
            }
            for i in range(bsz)
        ])

    @property
    def supports_prefetch(self):
        return super().supports_prefetch and getattr(self.tgt_factors, 'supports_prefetch', False)

    def prefetch(self, indices):
        super().prefetch(indices)
        self.tgt_factors.prefetch(indices)
