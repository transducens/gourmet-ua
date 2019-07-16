import math
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
import torch
from torch.distributions.bernoulli import Bernoulli
from fairseq import utils

@register_criterion('label_smoothed_cross_entropy_two_decoders')
class LabelSmoothedCrossEntropyTwoDecodersCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.b_weight = args.b_decoder_weight
        self.debug=args.debug_loss
        self.ignore_tags_distr=  Bernoulli(torch.tensor([args.tags_dropout]))

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        # fmt: off
        parser.add_argument('--b-decoder-weight', default=0.5, type=float,
                            help='Weight of auxiliary decoder in the loss')
        parser.add_argument('--tags-dropout', default=0.0, type=float,
                            help='Not optimize the tags output of a minitatch with this probability')
        parser.add_argument('--debug-loss', action='store_true',
                            help='Show debug information about how loss is computed for each minibatch')

    def forward(self, model, sample, reduce=True, training=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output,net_output_b = model(**sample['net_input'])

        if self.debug:
            print("Net output size: {}".format(net_output[0].size()))
            print("Net output B size: {}".format(net_output_b[0].size()))
            print("Computing loss on surface forms with reference: (size {}) {}".format(model.get_targets(sample,net_output).size(),model.get_targets(sample,net_output)))
            print("Computing loss on factors with reference: (size {}) {}".format( model.get_target_factors(sample,net_output_b).size() ,model.get_target_factors(sample,net_output_b)))

        loss_a, nll_loss_a = self.compute_loss(model, net_output, sample, reduce=reduce)
        loss_b, nll_loss_b = self.compute_loss_factors(model, net_output_b, sample, reduce=reduce)

        #We might have problems here when the number of TL factors is different from the number of TL tokens

        ignoreTags=False
        if training:
            if self.ignore_tags_distr.sample()[0] == 1.0:
                ignoreTags=True

        loss=loss_a*(1-self.b_weight)*2
        nll_loss=nll_loss_a*(1-self.b_weight)*2

        if not ignoreTags:
            loss+=loss_b*self.b_weight*2
            nll_loss+=nll_loss_b*self.b_weight*2

        if self.args.sentence_avg:
            raise NotImplementedError
        sample_size = sample['target'].size(0) if self.args.sentence_avg else (sample['ntokens'] if not ignoreTags else sample['ntokens_a'])
        sample_size_a = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens_a']
        sample_size_b = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens_b']

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'loss_a': utils.item(loss_a.data) if reduce else loss_a.data,
            'loss_b': utils.item(loss_b.data) if reduce else loss_b.data,
            'nll_loss_a': utils.item(nll_loss_a.data) if reduce else nll_loss_a.data,
            'nll_loss_b': utils.item(nll_loss_b.data) if reduce else nll_loss_b.data,
            'ntokens': sample['ntokens'],
            'ntokens_a': sample['ntokens_a'],
            'ntokens_b': sample['ntokens_b'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'sample_size_a': sample_size_a,
            'sample_size_b': sample_size_b,
        }
        return loss, sample_size, logging_output

    def compute_loss_factors(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target_factors = model.get_target_factors(sample, net_output).view(-1, 1)
        non_pad_mask = target_factors.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target_factors)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        ntokens_a = sum(log.get('ntokens', 0) for log in logging_outputs)
        ntokens_b = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        sample_size_a = sum(log.get('sample_size_a', 0) for log in logging_outputs)
        sample_size_b = sum(log.get('sample_size_b', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'loss_a': sum(log.get('loss_a', 0) for log in logging_outputs) / sample_size_a / math.log(2) if sample_size_a > 0 else 0.,
            'nll_loss_a': sum(log.get('nll_loss_a', 0) for log in logging_outputs) / ntokens_a / math.log(2) if ntokens_b > 0 else 0.,
            'loss_b': sum(log.get('loss_b', 0) for log in logging_outputs) / sample_size_b / math.log(2) if sample_size_b > 0 else 0.,
            'nll_loss_b': sum(log.get('nll_loss_b', 0) for log in logging_outputs) / ntokens_b / math.log(2) if ntokens_b > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
