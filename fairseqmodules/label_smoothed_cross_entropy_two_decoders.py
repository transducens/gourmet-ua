from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

@register_criterion('label_smoothed_cross_entropy_two_decoders')
class LabelSmoothedCrossEntropyTwoDecodersCriterion(LabelSmoothedCrossEntropyCriterion):
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output,net_output_b = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        loss_b, nll_loss_b = self.compute_loss_factors(model, net_output_b, sample, reduce=reduce)

        #We might have problems here when the number of TL factors is different from the number of TL tokens
        loss=loss+loss_b
        nll_loss=nll_loss+nll_loss_b

        if self.args.sentence_avg:
            raise NotImplementedError
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']


        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss_factors(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target_factors = model.get_target_factors(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target_factors)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss
