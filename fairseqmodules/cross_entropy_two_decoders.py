import math
from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion

from fairseq import utils

@register_criterion('cross_entropy_two_decoders')
class CrossEntropyTwoDecodersCriterion(CrossEntropyCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.b_weight = args.b_decoder_weight
        self.debug=args.debug_loss

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        # fmt: off
        parser.add_argument('--b-decoder-weight', default=0.5, type=float,
                            help='Weight of auxiliary decoder in the loss')
        parser.add_argument('--debug-loss', action='store_true',
                            help='Show debug information about how loss is computed for each minibatch')

    def forward(self, model, sample, reduce=True):
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

        loss_a, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        loss_b, _ = self.compute_loss_factors(model, net_output_b, sample, reduce=reduce)

        #We might have problems here when the number of TL factors is different from the number of TL tokens
        loss=loss_a*(1-self.b_weight)*2+loss_b*self.b_weight*2

        if self.args.sentence_avg:
            raise NotImplementedError
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        sample_size_a = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens_a']
        sample_size_b = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens_b']

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'loss_a': utils.item(loss_a.data) if reduce else loss_a.data,
            'loss_b': utils.item(loss_b.data) if reduce else loss_b.data,
            'ntokens': sample['ntokens'],
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
        loss=F.nll_loss(lprobs, target_factors, size_average=False, ignore_index=self.padding_idx,reduce=reduce)
        return loss, loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        sample_size_a = sum(log.get('sample_size_a', 0) for log in logging_outputs)
        sample_size_b = sum(log.get('sample_size_b', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'loss_a': sum(log.get('loss_a', 0) for log in logging_outputs) / sample_size_a / math.log(2) if sample_size_a > 0 else 0.,
            'loss_b': sum(log.get('loss_b', 0) for log in logging_outputs) / sample_size_b / math.log(2) if sample_size_b > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
