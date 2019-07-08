# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch

from fairseq import search, utils
from fairseq.models import FairseqIncrementalDecoder

from . import lstm_two_decoders_async_model,bahdanau_rnn_model

SPLITWORDMARK="@@"

class TwoDecoderAsyncBeamSearch(search.Search):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)
        self.tgt_dict=tgt_dict

    def step(self, step, lprobs, scores,tokens,sf_dict):
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            #TODO: we need to re-compute and normalize after this step

            #lprobs:(bsz x input_beam_size x vocab_size)
            #scores: (bsz x input_beam_size x step)
            #tokens: (bsz x input_beam_size x step)
            # scores are expanded so that the same score is added to all probs

            unnnorm_lprobs=(lprobs +scores[:, :, step - 1].unsqueeze(-1)).view(bsz,-1)

            #Now the last score is at position step-1
            pos_scores = scores.clone()[:,:, :step]

            #I think this is not needed:
            #pos_scores[:,:, step-1] = eos_scores

            # convert from cumulative to per-position scores
            pos_scores[:,:, 1:] = pos_scores[:,:, 1:] - pos_scores[:,:, :-1]
            pos_scores_sf=pos_scores[:,:,1::2]
            pos_scores_tags=pos_scores[:,:,0::2]

            #If the number of scores is odd, there is one additional tag
            num_sf=(step+1)//2

            #TODO: 2 dimension and broadcasting
            num_tags_pre_single=(step+1)-num_sf
            num_tags_pre= [ [ num_tags_pre_single for j in range(lprobs.size(1)) ] for i in range(lprobs.size(0)) ]

            #how does it change when we are in a tag step or on a sf step
            #how does it change when we are in a tag step with forced output?
            #
            #if we are in a tag step, and the last sf ends with @@: we do not take into account current tag
            # "                                       does not end with @@: we take into account current tag
            # if we are in a sf step: is the number of tags affected by the chosen sf? NO -> we can copy algorithm
            # from normalization

            #substract from numtags the number of non-end surface forms for each hypothesis
            tokens_sf=tokens[:,:,1::2]
            num_non_end=[      [  len( [ t for t in c if sf_dict[t].endswith(SPLITWORDMARK)  ]  )    for c in r]   for r in tokens_sf ]

            num_tags = torch.tensor(num_tags_pre,dtype=pos_scores.dtype,device=pos_scores.device)-torch.tensor(num_non_end,dtype=pos_scores.dtype,device=pos_scores.device)

            if TwoDecoderSequenceGenerator.DEBUG:
                print("Doing a beam search step")
                print("lprobs ({}): {}".format(lprobs.size(),lprobs))

            lprobs_add_sf=0.0
            lprobs_add_tags=0.0
            if self.tgt_dict == sf_dict:
                lprobs_add_sf=lprobs
            else:
                lprobs_add_tags=lprobs

            if TwoDecoderSequenceGenerator.DEBUG:
                print("pos_scores ({}): {}".format(pos_scores.size(),pos_scores))
                print("pos_scores_sf ({}): {}".format(pos_scores_sf.size(),pos_scores_sf))
                print("pos_scores_tags ({}): {}".format(pos_scores_tags.size(),pos_scores_tags))
                print("num_sf: {}".format(num_sf))
                print("tokens_sf ({}): {}".format(tokens_sf.size(),tokens_sf))
                print("num_tags_pre: {}".format(num_tags_pre))
                print("num_non_end: {}".format(num_non_end))
                print("num_tags  ({}): {}".format(num_tags.size(),num_tags))
                print("torch.sum(pos_scores_sf,-1).unsqueeze(1):{}".format(torch.sum(pos_scores_sf,-1).unsqueeze(-1).size()))

            lprobs_a=  (torch.sum(pos_scores_sf,-1).unsqueeze(-1) + lprobs_add_sf )/num_sf
            lprobs_b= (torch.sum(pos_scores_tags,-1).unsqueeze(-1) + lprobs_add_tags )/num_tags.unsqueeze(-1)

            if TwoDecoderSequenceGenerator.DEBUG:
                print("lprobs_a ({}): {}".format(lprobs_a.size(),lprobs_a))
                print("lprobs_b ({}): {}".format(lprobs_b.size(),lprobs_b))

            lprobs=  lprobs_a + lprobs_b

            if TwoDecoderSequenceGenerator.DEBUG:
                print("Final lprobs: {}".format(lprobs))

        torch.topk(
            #With view, we rearrange the original lprobs, which now has size=( bsz,  input_beam_size x vocab_size), i.e.,
            #we mix all the scores (coming from different hypotheses) for each batch element
            #And topk computes the highest scores for each batch element
            lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
            out=(self.scores_buf, self.indices_buf), #A namedtuple of (values, indices) is returned, where the indices are the indices of the elements in the original input tensor.
        )
        if step == 0:
            final_scores=self.scores_buf
        else:
            final_scores=torch.take(unnnorm_lprobs,self.indices_buf)
        torch.div(self.indices_buf, vocab_size, out=self.beams_buf)
        self.indices_buf.fmod_(vocab_size)
        if TwoDecoderSequenceGenerator.DEBUG:
            print("scores_buf: {}".format(self.scores_buf))
            print("final_scores: {}".format(final_scores))
        return final_scores, self.indices_buf, self.beams_buf

class TwoDecoderSequenceGenerator(object):
    DEBUG=False
    def __init__(
        self,
        tgt_dict,
        tgt_dict_b,#Dictionary of auxiliary output, which is decoded before the main one
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        stop_early=True,
        normalize_scores=True,
        len_penalty=1.,
        unk_penalty=0.,
        retain_dropout=False,
        sampling=False,
        sampling_topk=-1,
        sampling_temperature=1.,
        diverse_beam_groups=-1,
        diverse_beam_strength=0.5,
        match_source_len=False,
        no_repeat_ngram_size=0,
        only_output_factors=False,
        separate_factors_sf_models=False
    ):
        """Generates translations of a given source sentence.
        Args:
            tgt_dict (~fairseq.data.Dictionary): target dictionary
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            stop_early (bool, optional): stop generation immediately after we
                finalize beam_size hypotheses, even though longer hypotheses
                might have better normalized scores (default: True)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            sampling (bool, optional): sample outputs instead of beam search
                (default: False)
            sampling_topk (int, optional): only sample among the top-k choices
                at each step (default: -1)
            sampling_temperature (float, optional): temperature for sampling,
                where values >1.0 produces more uniform sampling and values
                <1.0 produces sharper sampling (default: 1.0)
            diverse_beam_groups/strength (float, optional): parameters for
                Diverse Beam Search sampling
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """


        #pad, unk and eos have the same indexes in all dictionaries.
        # See: https://github.com/pytorch/fairseq/blob/v0.6.2/fairseq/data/dictionary.py
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.vocab_size_b = len(tgt_dict_b)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size

        self.tgt_dict=tgt_dict
        self.tgt_dict_b=tgt_dict_b

        self.only_output_factors=only_output_factors
        self.independent_factors_models=separate_factors_sf_models

        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'

        if sampling:
            self.search = search.Sampling(tgt_dict, sampling_topk, sampling_temperature)
        elif diverse_beam_groups > 0:
            self.search = search.DiverseBeamSearch(tgt_dict, diverse_beam_groups, diverse_beam_strength)
        elif match_source_len:
            self.search = search.LengthConstrainedBeamSearch(
                tgt_dict, min_len_a=1, min_len_b=0, max_len_a=1, max_len_b=0,
            )
        else:
            self.search = search.BeamSearch(tgt_dict)
            self.search_b = search.BeamSearch(tgt_dict_b)

    @torch.no_grad()
    def generate(
        self,
        models,
        sample,
        prefix_tokens=None,
        bos_token=None,
        forced_factors=None,
        forced_surface_forms=None,
        **kwargs
    ):
        """Generate a batch of translations.
        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            forced_factors: List[List]: factors to be printed in each batch element
        """
        model = EnsembleModel(models,self.tgt_dict,self.tgt_dict_b,self.independent_factors_models)
        if not self.retain_dropout:
            model.eval()

        if model.async:
            #overwrite search instances
            self.search = TwoDecoderAsyncBeamSearch(self.tgt_dict)
            self.search_b = TwoDecoderAsyncBeamSearch(self.tgt_dict_b)


        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }

        src_tokens = encoder_input['src_tokens']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size

        if TwoDecoderSequenceGenerator.DEBUG:
            print("Starting generate() with forced_factors: {}".format(forced_factors))

        #It is tricky to force factors when input batch size > 1
        #When only support forced_factors with first dimension == 1
        if forced_factors != None:
            assert bsz == 1
            forced_factors=forced_factors[0]

        if forced_surface_forms != None:
            assert bsz == 1
            forced_surface_forms = forced_surface_forms[0]

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )

        # compute the encoder output for each beam
        encoder_outs,encoder_outs_factors, encoder_outs_slfactors = model.forward_encoder(encoder_input)


        #For bsz=3, beam_size=5, max_len=20
        #tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()

        #One encoder out for each hypothesis
        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)
        encoder_outs_factors = model.reorder_encoder_out_factors(encoder_outs_factors, new_order)
        encoder_outs_slfactors = model.reorder_encoder_out_slfactors(encoder_outs_slfactors, new_order)

        # initialize buffers
        #new: Constructs a new tensor of the same data type as self tensor.

        #CHANGE: *2
        scores = src_tokens.new(bsz * beam_size, (max_len + 1)*2 ).float().fill_(0)
        scores_buf = scores.clone()

        #CHANGE: *2
        tokens = src_tokens.data.new(bsz * beam_size, (max_len + 2)*2 ).long().fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = bos_token or self.eos

        #CHANGE: addition
        tokens[:, 1] = bos_token or self.eos

        attn, attn_buf = None, None
        nonpad_idxs = None

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        worst_finalized = [{'idx': None, 'score': -math.inf} for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes

        # For bsz=3, beam_size=5, max_len=20
        #tensor([[ 0],
        #[ 5],
        #[10]])
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfinalized_scores=None):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                if self.stop_early or step == max_len or unfinalized_scores is None:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                best_unfinalized_score = unfinalized_scores[sent].max()
                if self.normalize_scores:
                    best_unfinalized_score /= max_len ** self.len_penalty
                if worst_finalized[sent]['score'] >= best_unfinalized_score:
                    return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.
            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.
            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
                unfinalized_scores: A vector containing scores for all
                    unfinalized hypotheses
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            #print("finalize_hypos: tokens_clone 1: {}".format(tokens_clone))
            #CHANGE: skip first two indexes
            tokens_clone = tokens_clone[:, 2:step + 3]  # skip the first index, which is EOS
            #print("finalize_hypos: tokens_clone 2: {}".format(tokens_clone))
            tokens_clone[:, step] = self.eos
            #print("finalize_hypos: tokens_clone 3: {}".format(tokens_clone))
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 2:step+3] if attn is not None else None

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                if TwoDecoderSequenceGenerator.DEBUG:
                    print("Normalizing scores in async setup. eos_scores: {} step +1: {}".format(eos_scores,step + 1))
                    print("pos_scores: {}".format(pos_scores))

                if model.async:
                    # If we are in async set-up, we normalize independently the surface form
                    # and the additional tag channels
                    # We remove from the additional tag channel the time steps when
                    # the decoder was frozen
                    pos_scores_sf=pos_scores[:,1::2]
                    pos_scores_tags=pos_scores[:,0::2]

                    #If the number of scores is odd, there is one additional tag
                    num_sf=(step+1)//2
                    num_tags_pre=[ (step+1)-num_sf for i in range(eos_scores.numel()) ]

                    #substract from numtags the number of non-end surface forms for each hypothesis
                    tokens_sf=tokens_clone[:,1::2]
                    num_non_end=[  len( [ t for t in r if self.tgt_dict[t].endswith(SPLITWORDMARK)  ]  )  for r in tokens_sf ]
                    num_tags = torch.tensor(num_tags_pre,dtype=pos_scores.dtype,device=pos_scores.device)-torch.tensor(num_non_end,dtype=pos_scores.dtype,device=pos_scores.device)

                    eos_scores=  torch.sum(pos_scores_sf,-1)/num_sf**self.len_penalty + torch.sum(pos_scores_tags,-1)/num_tags**self.len_penalty
                    if TwoDecoderSequenceGenerator.DEBUG:
                        print("pos_scores_sf: {}".format(pos_scores_sf))
                        print("pos_scores_tags: {}".format(pos_scores_tags))
                        print("num_sf: {}".format(num_sf))
                        print("num_tags_pre: {}".format(num_tags_pre))
                        print("num_non_end: {}".format(num_non_end))
                        print("num_tags: {}".format(num_tags))
                        print("Final eos_scores: {}".format(eos_scores))

                else:
                    eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                if self.match_source_len and step > src_lengths[unfin_idx]:
                    score = -math.inf

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i][nonpad_idxs[sent]]
                        _, alignment = hypo_attn.max(dim=0)
                    else:
                        hypo_attn = None
                        alignment = None
                    if TwoDecoderSequenceGenerator.DEBUG:
                        print("Finalizing hypothesis: {}. Score: {} Tags: {}. Surface forms: {}.".format(tokens_clone[i], score, self.tgt_dict_b.string(tokens_clone[i][0::2]) , self.tgt_dict.string(tokens_clone[i][1::2])))

                    return {
                        'tokens': tokens_clone[i][1::2],
                        'tags': tokens_clone[i][0::2],
                        'raw_tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': alignment,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                elif not self.stop_early and score > worst_finalized[sent]['score']:
                    # replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sent]['idx']
                    if worst_idx is not None:
                        finalized[sent][worst_idx] = get_hypo()

                    # find new worst finalized hypo for this sentence
                    idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                    worst_finalized[sent] = {
                        'score': s['score'],
                        'idx': idx,
                    }

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfinalized_scores):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None
        #CHANGE: max_len*2
        for step in range((max_len + 1)*2):  # two extra steps for EOS markers
            is_decoder_b_step=False
            if step % 2 == 0:
                is_decoder_b_step=True

            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)

                model.reorder_incremental_state(reorder_state)

                encoder_outs=model.reorder_encoder_out(encoder_outs, reorder_state)
                encoder_outs_factors=model.reorder_encoder_out_factors(encoder_outs_factors, reorder_state)
                encoder_outs_slfactors=model.reorder_encoder_out_slfactors(encoder_outs_slfactors, reorder_state)

            if TwoDecoderSequenceGenerator.DEBUG:
                print("Incremental state: {}".format(model.incremental_states))
                print("Incremental state_b: {}".format(model.incremental_states_b))

            #Compute last scores chosen by beam search on the same decoder
            last_scores=[0.0 for i in range(scores.size(0))]
            if step <= 2:
                for i in range(scores.size(0)):
                    last_scores[i]=scores[i][0]
            else:
                for i in range(scores.size(0)):
                    last_scores[i]=scores[i][step-2] - scores[i][step-3]


            #CHANGE: call the appropriate decoder: step +1 -> step +2
            lprobs, avg_attn_scores = model.forward_decoder(tokens[:, :step + 2], encoder_outs,encoder_outs_factors,encoder_outs_slfactors,is_decoder_b_step,forced_factors=forced_factors,forced_surface_forms=forced_surface_forms,last_scores=last_scores)
            if is_decoder_b_step:
                d=self.tgt_dict_b
            else:
                d=self.tgt_dict
            if TwoDecoderSequenceGenerator.DEBUG:
                print("Results of the step\nMax lprobs: {} {} {}\n".format(torch.argmax(lprobs,-1), d.string(torch.argmax(lprobs,-1)),torch.max(lprobs,-1)))

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            if self.no_repeat_ngram_size > 0:
                raise NotImplementedError
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
                for bbsz_idx in range(bsz * beam_size):
                    gen_tokens = tokens[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                                gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            # Record attention scores
            if avg_attn_scores is not None:
                raise NotImplementedError
                if attn is None:
                    attn = scores.new(bsz * beam_size, src_tokens.size(1), max_len + 2)
                    attn_buf = attn.clone()
                    nonpad_idxs = src_tokens.ne(self.pad)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)
            #CHANGE: *2
            if step < max_len*2:
                self.search.set_src_lengths(src_lengths)

                if self.no_repeat_ngram_size > 0:
                    raise NotImplementedError
                    def calculate_banned_tokens(bbsz_idx):
                        # before decoding the next token, prevent decoding of ngrams that have already appeared
                        ngram_index = tuple(tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                        return gen_ngrams[bbsz_idx].get(ngram_index, [])

                    if step + 2 - self.no_repeat_ngram_size >= 0:
                        # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                        banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(bsz * beam_size)]
                    else:
                        banned_tokens = [[] for bbsz_idx in range(bsz * beam_size)]

                    for bbsz_idx in range(bsz * beam_size):
                        lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = -math.inf

                if prefix_tokens is not None and step < prefix_tokens.size(1):
                    raise NotImplementedError
                    probs_slice = lprobs.view(bsz, -1, lprobs.size(-1))[:, 0, :]
                    cand_scores = torch.gather(
                        probs_slice, dim=1,
                        index=prefix_tokens[:, step].view(-1, 1)
                    ).view(-1, 1).repeat(1, cand_size)
                    if step > 0:
                        # save cumulative scores for each hypothesis
                        cand_scores.add_(scores[:, step - 1].view(bsz, beam_size).repeat(1, 2))
                    cand_indices = prefix_tokens[:, step].view(-1, 1).repeat(1, cand_size)
                    cand_beams = torch.zeros_like(cand_indices)

                    # handle prefixes of different lengths
                    partial_prefix_mask = prefix_tokens[:, step].eq(self.pad)
                    if partial_prefix_mask.any():
                        partial_scores, partial_indices, partial_beams = self.search.step(
                            step,
                            lprobs.view(bsz, -1, self.vocab_size),
                            scores.view(bsz, beam_size, -1)[:, :, :step],
                        )
                        cand_scores[partial_prefix_mask] = partial_scores[partial_prefix_mask]
                        cand_indices[partial_prefix_mask] = partial_indices[partial_prefix_mask]
                        cand_beams[partial_prefix_mask] = partial_beams[partial_prefix_mask]
                else:
                    if TwoDecoderSequenceGenerator.DEBUG:
                        print("Calling beam search with\nscores:{}".format(scores.view(bsz, beam_size, -1)[:, :, :step]))
                    #CHANGE: search and search_b
                    search_f= self.search_b if is_decoder_b_step else self.search
                    my_vocab_size=self.vocab_size_b if is_decoder_b_step else self.vocab_size
                    if model.async:
                        cand_scores, cand_indices, cand_beams = search_f.step(
                            step,
                            lprobs.view(bsz, -1, my_vocab_size),
                            scores.view(bsz, beam_size, -1)[:, :, :step],
                            tokens.view(bsz, beam_size, -1)[:,:, :step + 2],
                            self.tgt_dict
                        )
                    else:
                        cand_scores, cand_indices, cand_beams = search_f.step(
                            step,
                            lprobs.view(bsz, -1, my_vocab_size),
                            scores.view(bsz, beam_size, -1)[:, :, :step]
                            )
                    if TwoDecoderSequenceGenerator.DEBUG:
                        print("Result of beam search\ncand_scores:{}\ncand_indices:{}\ncand_beams:{}\n".format( cand_scores, cand_indices, cand_beams))
            else:
                # make probs contain cumulative scores for each hypothesis
                lprobs.add_(scores[:, step - 1].unsqueeze(-1))

                # finalize all active hypotheses once we hit max_len
                # pick the hypothesis with the highest prob of EOS right now
                torch.sort(
                    lprobs[:, self.eos],
                    descending=True,
                    out=(eos_scores, eos_bbsz_idx),
                )
                num_remaining_sent -= len(finalize_hypos(step, eos_bbsz_idx, eos_scores))
                assert num_remaining_sent == 0
                break

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)

            finalized_sents = set()
            if step >= self.min_len:
                # only consider eos when it's among the top beam_size indices
                torch.masked_select(
                    cand_bbsz_idx[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_bbsz_idx,
                )
                if eos_bbsz_idx.numel() > 0:
                    torch.masked_select(
                        cand_scores[:, :beam_size],
                        mask=eos_mask[:, :beam_size],
                        out=eos_scores,
                    )
                    finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores, cand_scores)
                    num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len*2

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                    attn_buf.resize_as_(attn)
                bsz = new_bsz
            else:
                batch_idxs = None

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            active_mask = buffer('active_mask')
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(_ignore, active_hypos)
            )

            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            #CHANGE: +1 -> +2
            torch.index_select(
                tokens[:, :step + 2], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 2],
            )
            #CHANGE: +1 -> +2
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 2],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                    out=attn_buf[:, :, :step + 2],
                )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

            if TwoDecoderSequenceGenerator.DEBUG:
                print("End of loop.\nscores: {}\ntokens: {}\n".format(scores[:,:step+1],tokens[:,:step+3]))
            #print("Incremental state: {}".format(model.incremental_states))
            #print("Incremental state_b: {}".format(model.incremental_states_b))

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

            #Print linguistic factors
            if TwoDecoderSequenceGenerator.DEBUG:
                print( self.tgt_dict_b.string( finalized[sent][0]['tags']  ))

            if self.only_output_factors:
                sf_idx=finalized[sent][0]['tokens']
                tags_idx= finalized[sent][0]['tags']
                if model.async:
                    tags_idx=[ t for i,t in enumerate(tags_idx) if i < len(sf_idx) and  not self.tgt_dict.string( [ sf_idx[i] ]).endswith(SPLITWORDMARK)  ]
                print("TAGS: "+ self.tgt_dict_b.string( tags_idx  ))

        #Remove linguistic factors before returning
        return finalized


class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models,tgt_dict,tgt_dict_b,independent_factors_models=False):
        super().__init__()
        self.async=False
        if isinstance(models[0],lstm_two_decoders_async_model.LSTMTwoDecodersAsyncModel) or isinstance(models[0],  bahdanau_rnn_model.BahdanauRNNTwoDecodersAsyncModel):
            self.async=True

        self.surface_condition_tags=False
        self.tag_feedback_first_subword=False
        if (isinstance(models[0],bahdanau_rnn_model.BahdanauRNNTwoDecodersSyncModel) and isinstance(models[0].decoder_b,bahdanau_rnn_model.GRUDecoderTwoInputs)) or  isinstance(models[0],bahdanau_rnn_model.BahdanauRNNTwoDecodersMutualInfluenceAsyncModel) or (isinstance(models[0],bahdanau_rnn_model.BahdanauRNNTwoEncDecodersSyncModel) and isinstance(models[0].decoder_b,bahdanau_rnn_model.GRUDecoderTwoInputs)  ) :
            self.surface_condition_tags=True
            if isinstance(models[0],bahdanau_rnn_model.BahdanauRNNTwoDecodersMutualInfluenceAsyncModel):
                self.tag_feedback_first_subword=True

        if independent_factors_models:
            assert len(models) > 1
        self.independent_factors_models=independent_factors_models

        if self.independent_factors_models:
            #Even positions: factors
            #Odd positions: surface forms
            models_factors=models[0::2]
            models=models[1::2]

        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        self.incremental_states_b = None
        if all(isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.incremental_states = {m: {} for m in models}
        if all(isinstance(m.decoder_b, FairseqIncrementalDecoder) for m in models):
            self.incremental_states_b = {m: {} for m in models}

        self.models_factors=[]
        self.incremental_states_factors = None
        self.incremental_states_b_factors = None
        if self.independent_factors_models:
            self.models_factors = torch.nn.ModuleList(models_factors)
            if all(isinstance(m.decoder, FairseqIncrementalDecoder) for m in models_factors):
                self.incremental_states_factors = {m: {} for m in models_factors}
            if all(isinstance(m.decoder_b, FairseqIncrementalDecoder) for m in models_factors):
                self.incremental_states_b_factors = {m: {} for m in models_factors}

        self.tgt_dict=tgt_dict
        self.tgt_dict_b=tgt_dict_b

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        encoder_outs_slfactors=[]
        if 'src_factors' in encoder_input:
            encoder_outs_slfactors=[ model.encoder_b(encoder_input['src_factors'],encoder_input['src_factors_lengths']) for model in self.models ]
        return [model.encoder(**encoder_input) for model in self.models],[model.encoder(**encoder_input) for model in self.models_factors],encoder_outs_slfactors

    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs,encoder_outs_factors,encoder_outs_slfactors, is_decoder_b_step=False,forced_factors=None,forced_surface_forms=None,last_scores=None):
        if len(self.models)+len(self.models_factors) == 1:
            return self._decode_one(
                tokens,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                encoder_outs_slfactors[0] if self.has_encoder() and len(encoder_outs_slfactors) > 0 else None,
                self.incremental_states,
                log_probs=True,
                is_decoder_b_step=is_decoder_b_step,
                forced_factors=forced_factors,
                forced_surface_forms=forced_surface_forms,
                last_scores=last_scores,

            )

        #Not supported in ensemble mode
        assert len(encoder_outs_slfactors) == 0

        log_probs = []
        avg_attn = None

        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(tokens, model, encoder_out,None, self.incremental_states, log_probs=True,is_decoder_b_step=is_decoder_b_step,forced_factors=forced_factors,forced_surface_forms=forced_surface_forms,last_scores=last_scores)
            if not (self.independent_factors_models and is_decoder_b_step):
                log_probs.append(probs)
                if attn is not None:
                    if avg_attn is None:
                        avg_attn = attn
                    else:
                        avg_attn.add_(attn)
        if self.independent_factors_models:
            for model, encoder_out in zip(self.models_factors, encoder_outs_factors):
                probs, attn = self._decode_one(tokens, model, encoder_out,None, self.incremental_states, log_probs=True,is_decoder_b_step=is_decoder_b_step,forced_factors=forced_factors,forced_surface_forms=forced_surface_forms,last_scores=last_scores)
                if self.independent_factors_models and is_decoder_b_step:
                    log_probs.append(probs)
                    if attn is not None:
                        if avg_attn is None:
                            avg_attn = attn
                        else:
                            avg_attn.add_(attn)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(log_probs))
        if avg_attn is not None:
            avg_attn.div_(len(log_probs))
        return avg_probs, avg_attn

    def _decode_one(self, tokens, model, encoder_out, encoder_out_slfactors,  incremental_states_do_not_use_me, log_probs,is_decoder_b_step,forced_factors,forced_surface_forms,last_scores):
        if TwoDecoderSequenceGenerator.DEBUG:
            print("Starting _decode_one with forced_factors: {}".format(forced_factors))
        dummy_steps=[False for i in range(tokens.size(0))]
        forced_word_ids=None
        if is_decoder_b_step:
            dec = model.decoder_b
            dict_a=self.tgt_dict_b
            dict_b=self.tgt_dict
            #Factors decoder input: factors, surface forms
            tokens_in_a=torch.index_select(tokens, -1, torch.tensor(  [i for i in range(0,tokens.size(-1),2) ] ).to(tokens.device) )
            tokens_in_b=torch.index_select(tokens, -1, torch.tensor(  [i for i in range(1,tokens.size(-1),2) ] ).to(tokens.device))

            if self.async:
                dummy_steps=[False for i in range(tokens.size(0))]
                #If last element of tokens_b is not an end of word, activate dummy flag
                for i in range(tokens_in_b.size(0)):
                    if  self.tgt_dict[ tokens_in_b[i][-1] ].endswith(SPLITWORDMARK):
                        dummy_steps[i]=True

            #Count number of generated full surface forms for each hypothesis to decide the factor to force
            if forced_factors:
                forced_word_ids=[]
                for i in range(tokens_in_b.size(0)):
                    if self.async:
                        #TODO: review me
                        # tokens_in_b[i][1:] because first token is padding
                        cand_position=len( [t for t in tokens_in_b[i][1:] if not  self.tgt_dict[ t ].endswith(SPLITWORDMARK) ])
                    else:
                        #Forced factor index is just based on length of already generated factors
                        cand_position=tokens_in_a.size(1)-1
                    next_factor= forced_factors[cand_position] if cand_position < len(forced_factors) else self.tgt_dict_b.eos()
                    forced_word_ids.append(next_factor)

            #Async:
            #if last element of tokens_in_b is not an end of word:
            #   - restore previous state after calling the decoder
            #   - alter output probabilities: set same tag to 1 and others to 0 (be careful with logs!!)
            #else:
            #   - Nothing to do: morph tags are repeated
        else:
            #Async:
            #   - Nothing to do: morph tags are repeated


            dec = model.decoder
            dict_a=self.tgt_dict
            dict_b=self.tgt_dict_b
            #surface forms decoder input: surface forms, factors
            tokens_in_a=torch.index_select(tokens, -1, torch.tensor( [i for i in range(1,tokens.size(-1),2) ] ).to(tokens.device))
            tokens_in_b=torch.index_select(tokens, -1, torch.tensor( [i for i in range(0,tokens.size(-1),2) ] ).to(tokens.device))

            if forced_surface_forms:
                forced_word_ids=[]
                #Count number of generated surface forms fragments for each hypothesis to decide the surface form to force
                #[ The same one for all hypotheses ]
                for i in range(tokens_in_a.size(0)):
                    cand_position=tokens_in_a.size(1)-1
                    next_word= forced_surface_forms[cand_position] if cand_position < len(forced_surface_forms) else self.tgt_dict.eos()
                    forced_word_ids.append(next_word)

        if TwoDecoderSequenceGenerator.DEBUG:
            print("Doing one step in the decoder\nlast_scores:{}\ntokens:{}\nis_decoder_b_step: {}\ntokens_in_a: {}\ntokens_in_b:{}\ndummy steps: {}".format(last_scores,tokens,is_decoder_b_step,tokens_in_a,tokens_in_b, dummy_steps))
            print("words_in_a: {}\nwords_in_b: {}\n".format( [dict_a.string(ts) for ts in tokens_in_a ],  [dict_b.string(ts) for ts in tokens_in_b ] ))

        #TODO: think about whether I should change this condition
        if self.incremental_states is not None:
            #print("{} {}".format(self.async,is_decoder_b_step))
            if is_decoder_b_step:
                input_state=  self.incremental_states_b_factors[model] if model in self.models_factors else self.incremental_states_b[model]

                #This structure depends on the particular model and might not work
                #with models different from LSTM or multi-layer LSTM

                #List of dictionaries
                #Each dictionary: beam id -> state to keep
                backup_states=[]
                incremental_state_key=utils._get_full_incremental_state_key(dec, 'cached_state')
                #TODO: check whether this works with GRU model
                if self.async and incremental_state_key in input_state:
                    for state_comp_idx,state_comp in enumerate(input_state[incremental_state_key]):
                        if isinstance(state_comp,list):
                            state_comp=state_comp[0]
                        d={}
                        for beam_idx,beam_state in enumerate(state_comp):
                            if dummy_steps[beam_idx]:
                                d[beam_idx]=beam_state
                        backup_states.append(d)

                if TwoDecoderSequenceGenerator.DEBUG:
                    print("Backup states: {}".format(backup_states))
                if self.surface_condition_tags:
                    #The forward method of the decoder also selects the last token for each row
                    #If we are working with an async decoder with feedback, we manipulate
                    #the input and choose the first part of a BPEd word instead
                    tokens_in_b_input=tokens_in_b[:,-1:]
                    if self.tag_feedback_first_subword:
                        #tokens_in_b_input is not the last subword unit, but the last beginning of subword
                        for i in range(tokens_in_b_input.size(0)):
                            #Find the last beginning of word
                            last_word_end=None
                            word_ends=[ idx for idx,w in enumerate(tokens_in_b_input[i]) if not self.tgt_dict.string([w]).endswith(SPLITWORDMARK) ]
                            if len(word_ends) > 0:
                                last_word_end=word_ends[-1]
                            #last_word_end contains the position in tokens_in_b[i] of the last bpe piece that is a word end
                            #If last_word_end is None: no word has finished yet: feedback is first token
                            #If last_word_end is the last position of tokens_in_b[i]: it is the feedback
                            #Otherwise: the feedback is the token after last_word_end
                            #This works also with padding tokens (always the first one) as it is
                            #considered as a word end
                            if last_word_end is None:
                                tokens_in_b_input[i][0]=tokens_in_b[i][0]
                            elif last_word_end == len(tokens_in_b[i])-1:
                                tokens_in_b_input[i][0]=tokens_in_b[i][last_word_end]
                            else:
                                tokens_in_b_input[i][0]=tokens_in_b[i][last_word_end+1]

                    decoder_out = list(dec(tokens_in_a, tokens_in_b_input, encoder_out_slfactors if encoder_out_slfactors is not None else encoder_out, incremental_state=input_state))
                else:
                    decoder_out = list(dec(tokens_in_a, encoder_out_slfactors if encoder_out_slfactors is not None else encoder_out, incremental_state=input_state))

                #Restore states
                if self.async:
                    for state_comp_idx,state_comp_dict in enumerate(backup_states):
                        for k in state_comp_dict:
                            if isinstance(input_state[incremental_state_key][state_comp_idx],list):
                                input_state[incremental_state_key][state_comp_idx][0][k]=state_comp_dict[k]
                            else:
                                input_state[incremental_state_key][state_comp_idx][k]=state_comp_dict[k]
            else:
                decoder_out = list(dec(tokens_in_a,tokens_in_b, encoder_out, incremental_state= self.incremental_states_factors[model] if model in self.models_factors else self.incremental_states[model] ))
        else:
            if self.async and is_decoder_b_step:
                decoder_out = list(dec(tokens_in_a, encoder_out_slfactors if encoder_out_slfactors is not None else encoder_out))
            else:
                decoder_out = list(dec(tokens_in_a,tokens_in_b, encoder_out_slfactors if encoder_out_slfactors is not None and is_decoder_b_step else encoder_out))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn['attn']
        if attn is not None:
            if type(attn) is dict:
                attn = attn['attn']
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]

        for i,is_dummy in enumerate(dummy_steps):
            if is_dummy:
                #Force decoder to produce the same tag
                probs[i][:]=-math.inf
                probs[i][ tokens_in_a[i][-1]  ]=0.0#last_scores[i]


        if forced_word_ids:
            if TwoDecoderSequenceGenerator.DEBUG:
                print("Forcing the folowwing word ids: {}".format(forced_word_ids))
            for i,fid in enumerate(forced_word_ids):
                probs[i][:]=-math.inf
                probs[i][fid]=0.0

        return probs, attn

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]
    def reorder_encoder_out_factors(self, encoder_outs_factors, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models_factors, encoder_outs_factors)
        ]
    def reorder_encoder_out_slfactors(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder_b.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            return
        for model in self.models:
            model.decoder.reorder_incremental_state(self.incremental_states[model], new_order)

        if self.incremental_states_b != None:
            for model in self.models:
                model.decoder_b.reorder_incremental_state(self.incremental_states_b[model], new_order)

        for model in self.models_factors:
            model.decoder.reorder_incremental_state(self.incremental_states_factors[model], new_order)

        if self.incremental_states_b_factors != None:
            for model in self.models_factors:
                model.decoder_b.reorder_incremental_state(self.incremental_states_b_factors[model], new_order)
