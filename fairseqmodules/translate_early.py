import fairseq.tasks.translation
from fairseq.tasks import register_task

@register_task('translation_early')
class TranslationEarlyStopTask(fairseq.tasks.translation.TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.losses=[]

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        self.losses.append(loss)
        print(self.losses)
