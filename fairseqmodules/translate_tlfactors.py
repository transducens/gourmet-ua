import sys

from fairseq.tasks import register_task
from fairseq.meters import AverageMeter

from . import translate_early
from . import language_pair_tl_factors_dataset

@register_task('translation_tlfactors')
class TranslationTLFactorsTask(translate_early.TranslationEarlyStopTask):

    @staticmethod
    def load_pretrained_model(path, src_dict_path, tgt_dict_path , tgt_factors_dict_path, arg_overrides=None):
        model = utils.load_checkpoint_to_cpu(path)
        args = model['args']
        state_dict = model['model']
        args = utils.override_model_args(args, arg_overrides)
        src_dict = Dictionary.load(src_dict_path)
        tgt_dict = Dictionary.load(tgt_dict_path)
        tgt_factors_dict = Dictionary.load(tgt_factors_dict_path)
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        task = TranslationTLFactorsTask(args, src_dict, tgt_dict,tgt_factors_dict)
        model = task.build_model(args)
        model.upgrade_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
        return model

    def __init__(self, args, src_dict, tgt_dict, tgt_factors_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.tgt_factors_dict=tgt_factors_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        parent_task= super(TranslationTLFactorsTask, cls).setup_task(args,kwargs)
        tgt_factors_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}factors.txt'.format(args.target_lang)))
        return  cls(args, parent_task.src_dict, parent_task.tgt_dict, tgt_factors_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                if self.args.lazy_load:
                    return IndexedDataset(path, fix_lua_indexing=True)
                else:
                    return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        src_datasets = []
        tgt_datasets = []
        tgt_factors_datasets = []

        data_paths = self.args.data

        for dk, data_path in enumerate(data_paths):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                src_datasets.append(indexed_dataset(prefix + src, self.src_dict))
                tgt_datasets.append(indexed_dataset(prefix + tgt, self.tgt_dict))
                tgt_factors_datasets.append(indexed_dataset(prefix + tgt + "factors", self.tgt_factors_dict))

                print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets[-1])))

                if not combine:
                    break

        assert len(src_datasets) == len(tgt_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset , tgt_factors_dataset = src_datasets[0], tgt_datasets[0], tgt_factors_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
            tgt_factors_dataset = ConcatDataset(tgt_factors_datasets, sample_ratios)

        self.datasets[split] = language_pair_tl_factors_dataset.LanguagePairTLFactorsDataset(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            tgt_factors_dataset, tgt_factors_dataset.sizes, self.tgt_factors_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

     @property
    def target_factors_dictionary(self):
        """Return the target factors :class:`~fairseq.data.Dictionary`."""
        return self.tgt_factors_dict