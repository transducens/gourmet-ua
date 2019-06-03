import sys

from fairseq.tasks import register_task
from fairseq.meters import AverageMeter

from . import translate_early 

@register_task('translation_tlfactors')
class TranslationTLFactorsTask(translate_early.TranslationEarlyStopTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
