#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

from collections import Counter
from itertools import zip_longest

from fairseq import options, tasks
from fairseq.data import indexed_dataset
from fairseq.binarizer import Binarizer
from fairseq.utils import import_user_module
from multiprocessing import Pool

import os
import shutil
import tempfile

def main(args):
    import_user_module(args)

    print(args)

    os.makedirs(args.destdir, exist_ok=True)
    target = not args.only_source

    task = tasks.get_task(args.task)

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def remove_interleaving_tags(infile,outfile,only_first_subword=False,only_last_subword=False):
        with open(infile) as in_f, open(outfile,"w") as out_f:
            for l in in_f:
                l=l.rstrip("\n")
                toks=l.split()
                outstr=" ".join([  t for i,t in enumerate(toks) if not t.startswith("interleaved_") and (toks[i-1].startswith("interleaved_") or not only_first_subword  and ( not t.endswith("@@")  or not only_last_subword) ])
                out_f.write(outstr)
                out_f.write("\n")

    def retain_interleaving_tags(infile,outfile,add_mark=True, match_bpe=True):
        NOEOW="@@"
        with open(infile) as in_f, open(outfile,"w") as out_f:
            for l in in_f:
                l=l.rstrip("\n")
                toks=l.split()
                words=[  t for t in toks if not t.startswith("interleaved_")]
                tags=[  t for t in toks if t.startswith("interleaved_")]
                o=[]
                tagp=0
                if match_bpe:
                    for w in words:
                        if not w.endswith(NOEOW):
                            #This is the end of a word
                            o.append(tags[tagp])
                            tagp+=1
                        else:
                            #This is not
                            o.append(tags[tagp]+(NOEOW if add_mark else ""))
                else:
                    o=tags
                outstr=" ".join(o)
                out_f.write(outstr)
                out_f.write("\n")

    def build_dictionary(filenames, src=False, tgt=False, factors=False, add_mark=True):
        assert src ^ tgt

        in_filenames=filenames
        temp_filenames=None
        if args.additional_decoder_tl:
            temp_filenames=set()
            for fn in filenames:
                tmpf=tempfile.NamedTemporaryFile(delete=False)
                tmpfn=tmpf.name
                tmpf.close()
                if factors == True:
                    retain_interleaving_tags(fn,tmpfn,add_mark)
                else:
                    remove_interleaving_tags(fn,tmpfn)
                temp_filenames.add(tmpfn)
            in_filenames=temp_filenames

        print("Building dictionaries from {}".format(in_filenames))
        rvalue= task.build_dictionary(
            in_filenames,
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
        )

        if temp_filenames != None:
            for fn in temp_filenames:
                os.remove(fn)

        return rvalue


    if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
        raise FileExistsError(dict_path(args.source_lang))
    if target and not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
        raise FileExistsError(dict_path(args.target_lang))

    if args.joined_dictionary:
        assert not args.srcdict or not args.tgtdict, \
            "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        elif args.tgtdict:
            src_dict = task.load_dictionary(args.tgtdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary(
                {train_path(lang) for lang in [args.source_lang, args.target_lang]}, src=True
            )
        tgt_dict = src_dict
    else:
        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary([train_path(args.source_lang)], src=True)

        if target:
            if args.tgtdict:
                tgt_dict = task.load_dictionary(args.tgtdict)
            else:
                assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = build_dictionary([train_path(args.target_lang)], tgt=True)
        else:
            tgt_dict = None

    src_dict.save(dict_path(args.source_lang))
    if target and tgt_dict is not None:
        tgt_dict.save(dict_path(args.target_lang))

    #If we are using a second decoder, we need an independent dictionary
    if args.additional_decoder_tl:
        tgt_factors_dict= build_dictionary([train_path(args.target_lang)], tgt=True,factors=True, add_mark=not args.disable_bpe_marks)
        tgt_factors_dict.save(dict_path(args.target_lang+"factors"))

    #If the SL text also contains linguistic factors, create a dictionary for them
    if args.source_factors:
        src_factors_dict=build_dictionary([train_path(args.source_lang)], src=True,tgt=False,factors=True, add_mark=False)
        src_factors_dict.save(dict_path(args.source_lang+"factors"))

    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers):
        print("| [{}] Dictionary: {} types".format(lang, len(vocab) - 1))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )

        input_temp_file=None
        if args.additional_decoder_tl:
            input_temp_file_obj=tempfile.NamedTemporaryFile(delete=False)
            input_temp_file_obj.close()
            input_temp_file=input_temp_file_obj.name
            if not output_prefix.endswith("factors"):
                remove_interleaving_tags(input_file,input_temp_file,only_first_subword= output_prefix.endswith("firstsubword"),only_last_subword= output_prefix.endswith("lastsubword"))
            else:
                if output_prefix.endswith("asyncfactors"):
                    retain_interleaving_tags(input_file,input_temp_file,add_mark=False, match_bpe=False)
                else:
                    retain_interleaving_tags(input_file,input_temp_file,add_mark=not args.disable_bpe_marks)
            input_file=input_temp_file


        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        vocab,
                        prefix,
                        lang,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                    callback=merge_result
                )
            pool.close()

        ds = indexed_dataset.IndexedDatasetBuilder(
            dataset_dest_file(args, output_prefix, lang, "bin")
        )
        merge_result(
            Binarizer.binarize(
                input_file, vocab, lambda t: ds.add_item(t),
                offset=0, end=offsets[1]
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        print(
            "| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
            )
        )

        if input_temp_file != None:
            os.remove(input_temp_file)

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args.output_format == "binary":
            make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers)
        elif args.output_format == "raw":
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)

    def make_all(lang, vocab, factors=False, async_factors=False, only_first_subword=False, only_last_subword=False):
        assert not (only_first_subword and only_last_subword)
        prefsuf=""
        if factors:
            if async_factors:
                prefsuf="asyncfactors"
            else:
                prefsuf="factors"
        if only_first_subword and not only_last_subword:
            prefsuf="firstsubword"
        if only_last_subword and not only_first_subword:
            prefsuf="lastsubword"
        if args.trainpref:
            make_dataset(vocab, args.trainpref, "train{}".format(prefsuf), lang, num_workers=args.workers)
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}{}".format(k,prefsuf) if k > 0 else "valid{}".format(prefsuf)
                make_dataset(vocab, validpref, outprefix, lang, num_workers=args.workers)
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}{}".format(k,prefsuf) if k > 0 else "test{}".format(prefsuf)
                make_dataset(vocab, testpref, outprefix, lang, num_workers=args.workers)

    make_all(args.source_lang, src_dict)
    if args.source_factors:
        #For consistency, we will call the file "asyncfactors"
        #Moreoveer, make_binary_dataset will repeat interleaved tags to match bpe
        #unles the file ends in "asyncfactors"
        make_all(args.source_lang, src_factors_dict,factors=True,async_factors=True)
    if target:
        make_all(args.target_lang, tgt_dict)
        if args.additional_decoder_tl:
            make_all(args.target_lang, tgt_factors_dict, factors=True)
            if args.disable_bpe_marks:
                #Create an additional version in which tgt factors are not duplicated to match bpe
                make_all(args.target_lang, tgt_factors_dict, factors=True, async_factors=True)
                if args.async_tags_surface_feed_first:
                    #Create an additional version that contains only the first subword of each TL word
                    make_all(args.target_lang, tgt_factors_dict, only_first_subword=True)
                if args.async_tags_surface_feed_last:
                    #Create an additional version that contains only the first subword of each TL word
                    make_all(args.target_lang, tgt_factors_dict, only_last_subword=True)
    print("| Wrote preprocessed data to {}".format(args.destdir))

    if args.alignfile:
        assert args.trainpref, "--trainpref must be set if --alignfile is specified"
        src_file_name = train_path(args.source_lang)
        tgt_file_name = train_path(args.target_lang)
        freq_map = {}
        with open(args.alignfile, "r", encoding='utf-8') as align_file:
            with open(src_file_name, "r", encoding='utf-8') as src_file:
                with open(tgt_file_name, "r", encoding='utf-8') as tgt_file:
                    for a, s, t in zip_longest(align_file, src_file, tgt_file):
                        si = src_dict.encode_line(s, add_if_not_exist=False)
                        ti = tgt_dict.encode_line(t, add_if_not_exist=False)
                        ai = list(map(lambda x: tuple(x.split("-")), a.split()))
                        for sai, tai in ai:
                            srcidx = si[int(sai)]
                            tgtidx = ti[int(tai)]
                            if srcidx != src_dict.unk() and tgtidx != tgt_dict.unk():
                                assert srcidx != src_dict.pad()
                                assert srcidx != src_dict.eos()
                                assert tgtidx != tgt_dict.pad()
                                assert tgtidx != tgt_dict.eos()

                                if srcidx not in freq_map:
                                    freq_map[srcidx] = {}
                                if tgtidx not in freq_map[srcidx]:
                                    freq_map[srcidx][tgtidx] = 1
                                else:
                                    freq_map[srcidx][tgtidx] += 1

        align_dict = {}
        for srcidx in freq_map.keys():
            align_dict[srcidx] = max(freq_map[srcidx], key=freq_map[srcidx].get)

        with open(
                os.path.join(
                    args.destdir,
                    "alignment.{}-{}.txt".format(args.source_lang, args.target_lang),
                ),
                "w", encoding='utf-8'
        ) as f:
            for k, v in align_dict.items():
                print("{} {}".format(src_dict[k], tgt_dict[v]), file=f)


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=True):
    ds = indexed_dataset.IndexedDatasetBuilder(
        dataset_dest_file(args, output_prefix, lang, "bin")
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, vocab, consumer, append_eos=append_eos,
                             offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    lang_part = (
        ".{}-{}.{}".format(args.source_lang, args.target_lang, lang) if lang is not None else ""
    )
    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def get_offsets(input_file, num_workers):
    return Binarizer.find_offsets(input_file, num_workers)


def merge_files(files, outpath):
    ds = indexed_dataset.IndexedDatasetBuilder("{}.bin".format(outpath))
    for file in files:
        ds.merge_file_(file)
        os.remove(indexed_dataset.data_file_path(file))
        os.remove(indexed_dataset.index_file_path(file))
    ds.finalize("{}.idx".format(outpath))


def cli_main():
    parser = options.get_preprocessing_parser()
    #Custom options
    parser.add_argument('--additional_decoder_tl', action='store_true',help='Add an additional decoder instead of interleaving')
    parser.add_argument('--disable_bpe_marks', action='store_true',help='Disable BPE marks on factors')
    parser.add_argument('--async_tags_surface_feed_first', action='store_true',help='Tags async decoder will receive the first subword of the previously generated surface form.')
    parser.add_argument('--async_tags_surface_feed_last', action='store_true',help='Tags async decoder will receive the last subword of the previously generated surface form.')
    parser.add_argument('--source-factors', action='store_true',help='SL text contains interleaved factors')
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
