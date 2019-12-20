#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput
import sys

import torch

from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator
from fairseq.utils import import_user_module



Batch = namedtuple('Batch', 'ids src_tokens src_lengths src_factors src_factors_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions):
    if hasattr(task,'source_factors_dictionary') and  task.source_factors_dictionary:
        lines_sf= [" ".join([t for t in src_str.split() if not t.startswith("interleaved_") ])
        for src_str in lines]
        lines_factors=[" ".join([t for t in src_str.split() if t.startswith("interleaved_") ])
        for src_str in lines]
        tokens = [
            task.source_dictionary.encode_line(src_str, add_if_not_exist=False).long()
            for src_str in lines_sf
        ]
        tokens_factors = [
            task.source_factors_dictionary.encode_line(src_str, add_if_not_exist=False).long()
            for src_str in lines_factors
        ]
        lengths = torch.LongTensor([t.numel() for t in tokens])
        lengths_factors=torch.LongTensor([t.numel() for t in tokens_factors])
        #print("Making batches from lines_sf: {}".format( lines_sf))
        #print("Making batches from lines_factors: {}".format( lines_factors))

    else:
        tokens = [
            task.source_dictionary.encode_line(src_str, add_if_not_exist=False).long()
            for src_str in lines
        ]
        tokens_factors=None
        lengths = torch.LongTensor([t.numel() for t in tokens])
        lengths_factors=None

    try:
        ds=task.build_dataset_for_inference(tokens, lengths,tokens_factors,lengths_factors)
    except TypeError:
        ds=task.build_dataset_for_inference(tokens, lengths)
    itr = task.get_batch_iterator(
        dataset=ds,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
            src_factors=batch['net_input']['src_factors'] if 'src_factors' in batch['net_input'] else None,
            src_factors_lengths=batch['net_input']['src_factors_lengths'] if 'src_factors_lengths' in batch['net_input'] else None
        )


def main(args):
    import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = utils.load_ensemble_for_inference(
        args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides),
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(args)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    start_id = 0
    for inputs in buffered_read(args.input, args.buffer_size):
        results = []
        for batch in make_batches(inputs, args, task, max_positions):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            src_factors=batch.src_factors
            src_factors_lengths=batch.src_factors_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()


            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }

            if src_factors is not None and src_factors_lengths is not None:
                src_factors = src_factors.cuda()
                src_factors_lengths = src_factors_lengths.cuda()

                sample['net_input']['src_factors']=src_factors
                sample['net_input']['src_factors_lengths']=src_factors_lengths


            translations = task.inference_step(generator, models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                print('S-{}\t{}'.format(id, src_str))

            # Process top predictions
            for hypo in hypos[:min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )

                #Re-interleave tags if needed
                if 'tags' in hypo:
                    hypo_tags_str=task.target_factors_dictionary.string(hypo['tags'].int().cpu())
                    fulltags=[ t for t in hypo_tags_str.split() if not t.endswith("@@")]
                    hypo_sfs=hypo_str.split()
                    new_hypo_l=[]
                    tags_count=0
                    for sf in hypo_sfs:
                        if (len(new_hypo_l) == 0 and len(fulltags) > 0) or (tags_count < len(fulltags) and not new_hypo_l[-1].endswith("@@") ):
                            new_hypo_l.append(fulltags[tags_count])
                            tags_count+=1
                        new_hypo_l.append(sf)
                    hypo_str=" ".join(new_hypo_l)

                print('H-{}\t{}\t{}'.format(id, hypo['score'], hypo_str))
                print('P-{}\t{}'.format(
                    id,
                    ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                ))
                if args.print_alignment:
                    print('A-{}\t{}'.format(
                        id,
                        ' '.join(map(lambda x: str(utils.item(x)), alignment))
                    ))

        # update running id counter
        start_id += len(results)


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
