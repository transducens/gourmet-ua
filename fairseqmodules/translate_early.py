import sys

import fairseq.tasks.translation
from fairseq.tasks import register_task
from fairseq.meters import AverageMeter

@register_task('translation_early')
class TranslationEarlyStopTask(fairseq.tasks.translation.TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.checkpoint_losses=[]
        self.valid_loss_meter=AverageMeter()
        self.after_valid_flag=False
        self.last_logging_output=None
        self.patience=args.early_stop_patience

        self.loss_field='loss'
        self.loss_sample_size_field=None

        assert not (args.early_stop_loss_a and args.early_stop_loss_b)

        if args.early_stop_loss_a:
            self.loss_field='loss_a'
            self.loss_sample_size_field='sample_size_a'
        elif args.early_stop_loss_b:
            self.loss_field='loss_b'
            self.loss_sample_size_field='sample_size_b'


    @staticmethod
    def add_args(parser):
        fairseq.tasks.translation.TranslationTask.add_args(parser)
        parser.add_argument('--early-stop-patience',  default=10, type=int,help='Early stop patience.')
        parser.add_argument('--early-stop-loss-a',  default=False, action='store_true',help='Early stop on loss of output a')
        parser.add_argument('--early-stop-loss-b',  default=False, action='store_true',help='Early stop on loss of output b')

    def valid_step(self, sample, model, criterion):
        self.after_valid_flag=True
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output

    def update_checkpoint_losses(self):
        if self.valid_loss_meter.count > 0:
            self.checkpoint_losses.append(self.valid_loss_meter.avg)
            print("Checkpoint losses: {}".format(self.checkpoint_losses))
            if len(self.checkpoint_losses) > self.patience:
                minloss=min(self.checkpoint_losses)
                if all( l > minloss for l in  self.checkpoint_losses[-self.patience:]  ):
                    print("Early stop")
                    sys.exit(0)

            self.valid_loss_meter.reset()

    def get_batch_iterator(self, dataset, max_tokens=None, max_sentences=None, max_positions=None,ignore_invalid_inputs=False, required_batch_size_multiple=1,seed=1, num_shards=1, shard_id=0, num_workers=0,epoch=0):
         self.update_checkpoint_losses()
         iterator=super().get_batch_iterator( dataset, max_tokens, max_sentences, max_positions,ignore_invalid_inputs, required_batch_size_multiple,seed, num_shards, shard_id, num_workers,epoch)
         return iterator

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        logging_output=super().aggregate_logging_outputs(logging_outputs, criterion)
        self.last_logging_output=logging_output
        return logging_output

    def grad_denom(self, sample_sizes, criterion):
        sample_size= super().grad_denom(sample_sizes, criterion)
        if self.after_valid_flag:
            #import pdb; pdb.set_trace()
            sample_size_for_early_stop=sample_size
            if self.loss_sample_size_field:
                sample_size_for_early_stop=self.last_logging_output.get(self.loss_sample_size_field, 0)
            self.valid_loss_meter.update(self.last_logging_output.get(self.loss_field, 0), sample_size_for_early_stop)
        self.after_valid_flag=False
        return sample_size
