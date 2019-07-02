import sys,os,itertools
import torch

from fairseq.tasks import register_task
from fairseq.meters import AverageMeter
from fairseq.data import (
    ConcatDataset,
    data_utils,
    Dictionary,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    LanguagePairDataset,
)

from . import translate_early


@register_task('translation_interleaving')
class TranslationInterleavingTask(translate_early.TranslationEarlyStopTask):
"""
Only needed to force factors during decoding with interleaving
"""
    @staticmethod
    def add_args(parser):
        translate_early.TranslationEarlyStopTask.add_args(parser)
        parser.add_argument('--force-factors',help='File that contains the factors that must be included in the output')

    def build_generator(self, args):
        if args.score_reference:
            raise NotImplementedError
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            #Load reference factors and convert them to arrays of numbers
            self.forced_factors=None
            if args.force_factors:
                self.forced_factors=[]
                #args.force_factors is a file
                with open(args.force_factors) as force_factors_f:
                    for line in force_factors_f:
                        line=line.rstrip("\n")
                        toks=line.split()
                        ids=[ self.target_factors_dictionary.index(t) for t in toks ]
                        self.forced_factors.append(ids)


            from . import forced_factors_sequence_generator
            return forced_factors_sequence_generator.ForcedFactorsSequenceGenerator(
                self.target_dictionary,
		beam_size=args.beam,
                max_len_a=args.max_len_a,
                max_len_b=args.max_len_b,
                min_len=args.min_len,
                stop_early=(not args.no_early_stop),
                normalize_scores=(not args.unnormalized),
                len_penalty=args.lenpen,
                unk_penalty=args.unkpen,
                sampling=args.sampling,
                sampling_topk=args.sampling_topk,
                sampling_temperature=args.sampling_temperature,
                diverse_beam_groups=args.diverse_beam_groups,
                diverse_beam_strength=args.diverse_beam_strength,
                match_source_len=args.match_source_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            batch_size=sample['net_input']['src_tokens'].size(0)
            input_forced_factors=None
            if self.forced_factors:
                input_forced_factors=self.forced_factors[:batch_size]
                self.forced_factors=self.forced_factors[batch_size:]
            return generator.generate(models, sample, prefix_tokens=prefix_tokens,forced_factors=input_forced_factors)
