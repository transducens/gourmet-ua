import fairseq.tasks.translation
from fairseq.tasks import register_task

@register_task('translation_early')
class TranslationEarlyStopTask(fairseq.tasks.translation.TranslationTask):
    pass
