import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.bernoulli import Bernoulli

from fairseq import options, utils
from fairseq.modules import AdaptiveSoftmax
from fairseq.models import (
    FairseqEncoder, FairseqIncrementalDecoder, BaseFairseqModel, register_model,
    register_model_architecture,
)
from fairseq.models.lstm import LSTMModel,LSTMEncoder,LSTMDecoder,base_architecture

@register_model('bahdanau_rnn')
class BahdanauRNNModel(LSTMModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        #Hack to call parent staticmethod
        LSTMModel.add_args(parser)
        parser.add_argument('--cond-gru', default=False, action='store_true',
                            help='Use conditional GRU as in Nematus')
        parser.add_argument('--debug', default=False, action='store_true',
                            help='Print content of minibacthes')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance.
        Encoders and decoders now are GRUs. Initialization of hidden state
        and output layers similar to Nematus
         """
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError('--encoder-layers must match --decoder-layers')

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = GRUEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
            debug=args.debug if 'debug' in args else False
        )
        decoder = GRUDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            cond_gru=args.cond_gru,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            debug=args.debug if 'debug' in args else False
        )
        r= cls(encoder, decoder)
        return r


@register_model('bahdanau_rnn_two_decoders_sync')
class BahdanauRNNTwoDecodersSyncModel(BahdanauRNNModel):
    def __init__(self, encoder, decoder, decoder_b):
        BaseFairseqModel.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_b = decoder_b
        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, GRUDecoderTwoInputs)
        assert isinstance(self.decoder_b, GRUDecoder) or isinstance(self.decoder_b, GRUDecoderTwoInputs)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        #Hack to call parent staticmethod
        BahdanauRNNModel.add_args(parser)
        parser.add_argument('--tags-condition-end', default=False, action='store_true',
                            help='Tags condition surface form decoder only at the end, as in lexical model')
        parser.add_argument('--surface-condition-tags', default=False, action='store_true',
                            help='Tag decoder has two inputs: previous timestep tag and previous timestep surface form')
        parser.add_argument('--share-embeddings-two-decoders', default=False, action='store_true',
                            help='Both decoders share embeddings')


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance.
        Encoders and decoders now are GRUs. Initialization of hidden state
        and output layers similar to Nematus
         """
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError('--encoder-layers must match --decoder-layers')

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            pretrained_decoder_embed_b = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            pretrained_decoder_embed_b = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
            if args.share_embeddings_two_decoders:
                if pretrained_decoder_embed is None:
                    pretrained_decoder_embed=Embedding(len(task.target_dictionary), args.decoder_embed_dim, task.target_dictionary.pad())
                pretrained_decoder_embed_b=Embedding(len(task.target_factors_dictionary),args.decoder_embed_dim,task.target_factors_dictionary.pad() )

        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = GRUEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
            debug=args.debug if 'debug' in args else False
        )

        decoder = GRUDecoderTwoInputs(
            dictionary=task.target_dictionary,
            dictionary_b=task.target_factors_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            pretrained_embed_b=pretrained_decoder_embed_b,
            share_input_output_embed=args.share_decoder_input_output_embed,
            b_condition_end=args.tags_condition_end,
            cond_gru=args.cond_gru if 'cond_gru' in args else False,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            debug=args.debug if 'debug' in args else False
        )

        if args.surface_condition_tags:
            decoder_b = GRUDecoderTwoInputs(
                dictionary=task.target_factors_dictionary,
                dictionary_b=task.target_dictionary,
                embed_dim=args.decoder_embed_dim,
                hidden_size=args.decoder_hidden_size,
                out_embed_dim=args.decoder_out_embed_dim,
                num_layers=args.decoder_layers,
                dropout_in=args.decoder_dropout_in,
                dropout_out=args.decoder_dropout_out,
                attention=options.eval_bool(args.decoder_attention),
                encoder_output_units=encoder.output_units,
                pretrained_embed=pretrained_decoder_embed_b,
                pretrained_embed_b=pretrained_decoder_embed,
                share_input_output_embed=args.share_decoder_input_output_embed,
                b_condition_end=args.tags_condition_end,
                cond_gru=args.cond_gru if 'cond_gru' in args else False,
                adaptive_softmax_cutoff=(
                    options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                    if args.criterion == 'adaptive_loss' else None
                ),
                debug=args.debug if 'debug' in args else False
            )
        else:
            decoder_b = GRUDecoder(
                dictionary=task.target_factors_dictionary,
                embed_dim=args.decoder_embed_dim,
                hidden_size=args.decoder_hidden_size,
                out_embed_dim=args.decoder_out_embed_dim,
                num_layers=args.decoder_layers,
                dropout_in=args.decoder_dropout_in,
                dropout_out=args.decoder_dropout_out,
                attention=options.eval_bool(args.decoder_attention),
                encoder_output_units=encoder.output_units,
                pretrained_embed=pretrained_decoder_embed_b,
                share_input_output_embed=args.share_decoder_input_output_embed,
                cond_gru=args.cond_gru if 'cond_gru' in args else False,
                adaptive_softmax_cutoff=(
                    options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                    if args.criterion == 'adaptive_loss' else None
                    ),
                debug=args.debug if 'debug' in args else False
        )

        r= cls(encoder, decoder, decoder_b)
        return r

    def get_target_factors(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target_factors']

    def forward(self, src_tokens, src_lengths, prev_output_tokens, prev_output_factors, cur_output_factors):
        """
        Run the forward pass for an encoder-decoder model.
        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::
            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
        Returns:
            the decoder's output, typically of shape `(batch, tgt_len, vocab)`
        """
        #print("Forward: prev_output_tokens:{}\nprev_output_factors:{}\ncur_output_factors:{}\n".format(prev_output_tokens, prev_output_factors, cur_output_factors))
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens,cur_output_factors, encoder_out)
        if isinstance(self.decoder_b,GRUDecoderTwoInputs):
            decoder_b_out = self.decoder_b(prev_output_factors,prev_output_tokens, encoder_out)
        else:
            decoder_b_out = self.decoder_b(prev_output_factors, encoder_out)
        return decoder_out, decoder_b_out

@register_model('bahdanau_rnn_two_encdecoders_sync')
class BahdanauRNNTwoEncDecodersSyncModel(BahdanauRNNModel):
    def __init__(self, encoder,encoder_b, decoder, decoder_b):
        BaseFairseqModel.__init__(self)
        self.encoder = encoder
        self.encoder_b = encoder_b
        self.decoder = decoder
        self.decoder_b = decoder_b
        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, GRUDecoderTwoInputs)
        assert isinstance(self.decoder_b, GRUDecoder) or isinstance(self.decoder_b, GRUDecoderTwoInputs)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        #Hack to call parent staticmethod
        BahdanauRNNModel.add_args(parser)
        parser.add_argument('--tags-condition-end', default=False, action='store_true',
                            help='Tags condition surface form decoder only at the end, as in lexical model')
        parser.add_argument('--surface-condition-tags', default=False, action='store_true',
                            help='Tag decoder has two inputs: previous timestep tag and previous timestep surface form')
        parser.add_argument('--share-embeddings-two-decoders', default=False, action='store_true',
                            help='Both decoders share embeddings')


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance.
        Encoders and decoders now are GRUs. Initialization of hidden state
        and output layers similar to Nematus
         """
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError('--encoder-layers must match --decoder-layers')

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )
        num_embeddings_b=len(task.source_factors_dictionary)
        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            pretrained_decoder_embed_b = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            pretrained_decoder_embed_b=None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
            if args.share_embeddings_two_decoders:
                if pretrained_decoder_embed is None:
                    pretrained_decoder_embed=Embedding(len(task.target_dictionary), args.decoder_embed_dim, task.target_dictionary.pad())
                pretrained_decoder_embed_b=Embedding(len(task.target_factors_dictionary),args.decoder_embed_dim,task.target_factors_dictionary.pad() )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = GRUEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
            debug=args.debug if 'debug' in args else False
        )

        encoder_b = GRUEncoder(
            dictionary=task.source_factors_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=Embedding(
                num_embeddings_b, args.encoder_embed_dim, task.source_factors_dictionary.pad()
            ),
            debug=args.debug if 'debug' in args else False
        )

        decoder = GRUDecoderTwoInputs(
            dictionary=task.target_dictionary,
            dictionary_b=task.target_factors_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            pretrained_embed_b=pretrained_decoder_embed_b,
            share_input_output_embed=args.share_decoder_input_output_embed,
            b_condition_end=args.tags_condition_end,
            cond_gru=args.cond_gru if 'cond_gru' in args else False,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            debug=args.debug if 'debug' in args else False
        )

        if args.surface_condition_tags:
            decoder_b = GRUDecoderTwoInputs(
                dictionary=task.target_factors_dictionary,
                dictionary_b=task.target_dictionary,
                embed_dim=args.decoder_embed_dim,
                hidden_size=args.decoder_hidden_size,
                out_embed_dim=args.decoder_out_embed_dim,
                num_layers=args.decoder_layers,
                dropout_in=args.decoder_dropout_in,
                dropout_out=args.decoder_dropout_out,
                attention=options.eval_bool(args.decoder_attention),
                encoder_output_units=encoder_b.output_units,
                pretrained_embed=pretrained_decoder_embed_b,
                pretrained_embed_b=pretrained_decoder_embed,
                share_input_output_embed=args.share_decoder_input_output_embed,
                b_condition_end=args.tags_condition_end,
                cond_gru=args.cond_gru if 'cond_gru' in args else False,
                adaptive_softmax_cutoff=(
                    options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                    if args.criterion == 'adaptive_loss' else None
                ),
                debug=args.debug if 'debug' in args else False
            )
        else:
            decoder_b = GRUDecoder(
                dictionary=task.target_factors_dictionary,
                embed_dim=args.decoder_embed_dim,
                hidden_size=args.decoder_hidden_size,
                out_embed_dim=args.decoder_out_embed_dim,
                num_layers=args.decoder_layers,
                dropout_in=args.decoder_dropout_in,
                dropout_out=args.decoder_dropout_out,
                attention=options.eval_bool(args.decoder_attention),
                encoder_output_units=encoder_b.output_units,
                pretrained_embed=pretrained_decoder_embed_b,
                share_input_output_embed=args.share_decoder_input_output_embed,
                cond_gru=args.cond_gru if 'cond_gru' in args else False,
                adaptive_softmax_cutoff=(
                    options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                    if args.criterion == 'adaptive_loss' else None
                    ),
                debug=args.debug if 'debug' in args else False
        )
        r= cls(encoder,encoder_b, decoder, decoder_b)
        return r

    def get_target_factors(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target_factors']

    def forward(self, src_tokens, src_lengths,src_factors,src_factors_lengths, prev_output_tokens, prev_output_factors, cur_output_factors):
        """
        Run the forward pass for an encoder-decoder model.
        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::
            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
        Returns:
            the decoder's output, typically of shape `(batch, tgt_len, vocab)`
        """

        #print("Forward: prev_output_tokens:{}\nprev_output_factors:{}\ncur_output_factors:{}\n".format(prev_output_tokens, prev_output_factors, cur_output_factors))
        encoder_out = self.encoder(src_tokens, src_lengths)
        encoder_b_out = self.encoder_b(src_factors, src_factors_lengths)
        decoder_out = self.decoder(prev_output_tokens,cur_output_factors, encoder_out)
        if isinstance(self.decoder_b,GRUDecoderTwoInputs):
            decoder_b_out = self.decoder_b(prev_output_factors,prev_output_tokens, encoder_b_out)
        else:
            decoder_b_out = self.decoder_b(prev_output_factors, encoder_b_out)
        return decoder_out, decoder_b_out

@register_model('bahdanau_rnn_two_decoders_async')
class BahdanauRNNTwoDecodersAsyncModel(BahdanauRNNModel):
    def __init__(self, encoder, decoder, decoder_b):
        BaseFairseqModel.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_b = decoder_b
        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, GRUDecoderTwoInputs)
        assert isinstance(self.decoder_b, GRUDecoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        #Hack to call parent staticmethod
        BahdanauRNNModel.add_args(parser)
        parser.add_argument('--tags-condition-end', default=False, action='store_true',
                            help='Tags condition surface form decoder only at the end, as in lexical model')
        parser.add_argument('--share-embeddings-two-decoders', default=False, action='store_true',
                            help='Both decoders share embeddings')


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance.
        Encoders and decoders now are GRUs. Initialization of hidden state
        and output layers similar to Nematus
         """
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError('--encoder-layers must match --decoder-layers')

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            pretrained_decoder_embed_b=pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            pretrained_decoder_embed_b = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
            if args.share_embeddings_two_decoders:
                if pretrained_decoder_embed is None:
                    pretrained_decoder_embed=Embedding(len(task.target_dictionary), args.decoder_embed_dim, task.target_dictionary.pad())
                pretrained_decoder_embed_b=Embedding(len(task.target_factors_dictionary),args.decoder_embed_dim,task.target_factors_dictionary.pad() )

        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = GRUEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
            debug=args.debug if 'debug' in args else False
        )

        decoder = GRUDecoderTwoInputs(
            dictionary=task.target_dictionary,
            dictionary_b=task.target_factors_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            pretrained_embed_b=pretrained_decoder_embed_b,
            share_input_output_embed=args.share_decoder_input_output_embed,
            b_condition_end=args.tags_condition_end,
            cond_gru=args.cond_gru if 'cond_gru' in args else False,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            debug=args.debug if 'debug' in args else False
        )

        decoder_b = GRUDecoder(
            dictionary=task.target_factors_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed_b,
            share_input_output_embed=args.share_decoder_input_output_embed,
            cond_gru=args.cond_gru if 'cond_gru' in args else False,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            debug=args.debug if 'debug' in args else False
        )
        r= cls(encoder, decoder, decoder_b)
        return r

    def get_target_factors(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target_factors_async']

    def forward(self, src_tokens, src_lengths, prev_output_tokens, prev_output_factors, cur_output_factors):
        """
        Run the forward pass for an encoder-decoder model.
        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::
            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
        Returns:
            the decoder's output, typically of shape `(batch, tgt_len, vocab)`
        """
        #print("Forward: prev_output_tokens:{}\nprev_output_factors:{}\ncur_output_factors:{}\n".format(prev_output_tokens, prev_output_factors, cur_output_factors))
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens,cur_output_factors, encoder_out)
        decoder_b_out = self.decoder_b(prev_output_factors, encoder_out)
        return decoder_out, decoder_b_out

@register_model('bahdanau_rnn_two_decoders_mutual_influence_async')
class BahdanauRNNTwoDecodersMutualInfluenceAsyncModel(BahdanauRNNModel):
    def __init__(self, encoder, decoder, decoder_b, feedback_encoder):
        BaseFairseqModel.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_b = decoder_b
        self.feedback_encoder= feedback_encoder
        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, GRUDecoderTwoInputs)
        assert isinstance(self.decoder_b, GRUDecoderTwoInputs)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        #Hack to call parent staticmethod
        BahdanauRNNModel.add_args(parser)
        parser.add_argument('--tags-condition-end', default=False, action='store_true',
                            help='Tags condition surface form decoder only at the end, as in lexical model')
        parser.add_argument('--feedback-encoder', default=False, action='store_true',
                            help='Use an encoder to condense all the previous surface forms.')
        parser.add_argument('--share-embeddings-two-decoders', default=False, action='store_true',
                            help='Both decoders share embeddings')


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance.
        Encoders and decoders now are GRUs. Initialization of hidden state
        and output layers similar to Nematus
         """
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError('--encoder-layers must match --decoder-layers')

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            pretrained_decoder_embed_b = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            pretrained_decoder_embed_b = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
            if args.share_embeddings_two_decoders:
                if pretrained_decoder_embed is None:
                    pretrained_decoder_embed=Embedding(len(task.target_dictionary), args.decoder_embed_dim, task.target_dictionary.pad())
                pretrained_decoder_embed_b=Embedding(len(task.target_factors_dictionary),args.decoder_embed_dim,task.target_factors_dictionary.pad() )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = GRUEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
            debug=args.debug if 'debug' in args else False
        )

        feedback_encoder=None
        if args.feedback_encoder:
            feedback_encoder = GRUEncoder(
                dictionary=task.target_dictionary,
                embed_dim=args.encoder_embed_dim,
                hidden_size=args.encoder_hidden_size,
                num_layers=args.encoder_layers,
                dropout_in=args.encoder_dropout_in,
                dropout_out=args.encoder_dropout_out,
                bidirectional=False,
                pretrained_embed=pretrained_decoder_embed,
                debug=args.debug if 'debug' in args else False
            )

        decoder = GRUDecoderTwoInputs(
            dictionary=task.target_dictionary,
            dictionary_b=task.target_factors_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            b_condition_end=args.tags_condition_end,
            cond_gru=args.cond_gru if 'cond_gru' in args else False,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            debug=args.debug if 'debug' in args else False
        )

        decoder_b = GRUDecoderTwoInputs(
            dictionary=task.target_factors_dictionary,
            dictionary_b=task.target_dictionary if not args.feedback_encoder else None, # empty dictionary means that we do not need to apply embeddings to second output
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            cond_gru=args.cond_gru if 'cond_gru' in args else False,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            debug=args.debug if 'debug' in args else False
        )
        r= cls(encoder, decoder, decoder_b, feedback_encoder)
        return r

    def get_target_factors(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target_factors_async']

    def forward(self, src_tokens, src_lengths, prev_output_tokens, prev_output_factors, cur_output_factors,prev_output_tokens_first_subword, prev_output_tokens_lengths):
        """
        Run the forward pass for an encoder-decoder model.
        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::
            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
        Returns:
            the decoder's output, typically of shape `(batch, tgt_len, vocab)`
        """
        #print("Forward: prev_output_tokens:{}\nprev_output_factors:{}\ncur_output_factors:{}\n".format(prev_output_tokens, prev_output_factors, cur_output_factors))

        if self.feedback_encoder is not None:
            second_input_decoder_b=self.feedback_encoder(prev_output_tokens, prev_output_tokens_lengths)
        else:
            second_input_decoder_b=prev_output_tokens_first_subword
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens,cur_output_factors, encoder_out)
        decoder_b_out = self.decoder_b(prev_output_factors,second_input_decoder_b, encoder_out)
        return decoder_out, decoder_b_out

@register_model('bahdanau_rnn_two_encdecoders_async')
class BahdanauRNNTwoEncDecodersAsyncModel(BahdanauRNNModel):
    def __init__(self, encoder,encoder_b, decoder, decoder_b):
        BaseFairseqModel.__init__(self)
        self.encoder = encoder
        self.encoder_b = encoder_b
        self.decoder = decoder
        self.decoder_b = decoder_b
        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, GRUDecoderTwoInputs)
        assert isinstance(self.decoder_b, GRUDecoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        #Hack to call parent staticmethod
        BahdanauRNNModel.add_args(parser)
        parser.add_argument('--tags-condition-end', default=False, action='store_true',
                            help='Tags condition surface form decoder only at the end, as in lexical model')
        parser.add_argument('--share-embeddings-two-decoders', default=False, action='store_true',
                            help='Both decoders share embeddings')


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance.
        Encoders and decoders now are GRUs. Initialization of hidden state
        and output layers similar to Nematus
         """
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError('--encoder-layers must match --decoder-layers')

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )
        num_embeddings_b=len(task.source_factors_dictionary)

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            pretrained_decoder_embed_b = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            pretrained_decoder_embed_b = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
            if args.share_embeddings_two_decoders:
                if pretrained_decoder_embed is None:
                    pretrained_decoder_embed=Embedding(len(task.target_dictionary), args.decoder_embed_dim, task.target_dictionary.pad())
                pretrained_decoder_embed_b=Embedding(len(task.target_factors_dictionary),args.decoder_embed_dim,task.target_factors_dictionary.pad() )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = GRUEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
            debug=args.debug if 'debug' in args else False
        )

        encoder_b = GRUEncoder(
            dictionary=task.source_factors_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=Embedding(
                num_embeddings_b, args.encoder_embed_dim, task.source_factors_dictionary.pad()
            ),
            debug=args.debug if 'debug' in args else False
        )

        decoder = GRUDecoderTwoInputs(
            dictionary=task.target_dictionary,
            dictionary_b=task.target_factors_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            pretrained_embed_b=pretrained_decoder_embed_b,
            share_input_output_embed=args.share_decoder_input_output_embed,
            b_condition_end=args.tags_condition_end,
            cond_gru=args.cond_gru if 'cond_gru' in args else False,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            debug=args.debug if 'debug' in args else False
        )

        decoder_b = GRUDecoder(
            dictionary=task.target_factors_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder_b.output_units,
            pretrained_embed=pretrained_decoder_embed_b,
            share_input_output_embed=args.share_decoder_input_output_embed,
            cond_gru=args.cond_gru if 'cond_gru' in args else False,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            debug=args.debug if 'debug' in args else False
        )
        r= cls(encoder,encoder_b, decoder, decoder_b)
        return r

    def get_target_factors(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target_factors_async']


    def forward(self, src_tokens, src_lengths ,src_factors,src_factors_lengths, prev_output_tokens, prev_output_factors, cur_output_factors):
        """
        Run the forward pass for an encoder-decoder model.
        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::
            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
        Returns:
            the decoder's output, typically of shape `(batch, tgt_len, vocab)`
        """
        #print("Forward: prev_output_tokens:{}\nprev_output_factors:{}\ncur_output_factors:{}\n".format(prev_output_tokens, prev_output_factors, cur_output_factors))
        encoder_out = self.encoder(src_tokens, src_lengths)
        encoder_b_out = self.encoder_b(src_factors, src_factors_lengths)
        decoder_out = self.decoder(prev_output_tokens,cur_output_factors, encoder_out)
        decoder_b_out = self.decoder_b(prev_output_factors, encoder_b_out)
        return decoder_out, decoder_b_out


class GRUEncoder(FairseqEncoder):
    """GRU encoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
            dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        left_pad=True, pretrained_embed=None, padding_value=0.,debug=False
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.debug=debug

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        #Here the LSTM overwrote the initialization of weights
        #We cannot apply dropout inside the GRU as Nematus
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out  if num_layers > 1 else 0., #This is ignored as it is not applied in single-layer GRUs
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        if self.debug:
            print("forward Encoder: ")
            print("src_lengths ({}): {}".format(src_lengths.size(),src_lengths))
            print("src_tokens ({}): {}".format(src_tokens.size(),src_tokens))
            print("")

        if self.left_pad:
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        #Like in nematus, dropout all the embedding dimensions of a word at the
        #same time
        #
        # Nematus uses tf.layers.dropout(x, noise_shape=(tf.shape(x)[0], tf.shape(x)[1], 1)
        # See https://pytorch.org/docs/stable/nn.html#dropout-layers and https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400/2
        # Reverted to 1d dropout
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist(),enforce_sorted=False)

        if self.debug:
            print("packed_x: {}".format(packed_x))
            print("")

        # apply GRU
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        packed_outs, final_hiddens = self.rnn(packed_x, h0)

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        if self.debug:
            print("unpacked x ({}): {}".format(x.size(),x))
            print("")

        # Since Nematus applies the same dropout mask to all timesteps inside the GRU,
        # we do the same here
        #
        # See https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400/2
        #
        # shape of x: (seq_len,bsz,hidden_size)
        #Reverted to 1d
        #x = x.permute(1, 0, 2)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        #x = x.permute(1, 0, 2)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            #shape: (num_layers, bsz, 2*hidden_size)


        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
        #shape: (seq_len,bsz)

        #print("Encoder: {}".format(self))
        #print("left_pad: {}".format(self.left_pad))
        #print("Intermediate encoder_padding_mask: {}".format(encoder_padding_mask))

        return {
            'encoder_out': (x, final_hiddens),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class ConcatAttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, alignment_dim, bias=True, dropout=0.0):
        super().__init__()

        self.input_proj = Linear(input_embed_dim+source_embed_dim, alignment_dim, bias=bias)
        self.activ_input_proj=nn.Tanh()
        self.score_proj = Linear(alignment_dim, 1, bias=bias)

        self.dropout_p=dropout
        self.dropout_input=nn.Dropout(dropout)

    def precompute_masked_source_hids(self,source_hids):
        #Apply same dropout to all timesteps of source_hids, like in Nematus
        mask = Bernoulli(torch.full_like(source_hids[0], 1 - self.dropout_p)).sample()/(1 - self.dropout_p)
        return source_hids*mask # In theory, with broadcasting we multiply all timesteps by the same mask

    def forward(self, input, masked_source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x source_embed_dim

        #Repeat TL hidden state srclen times
        # x: srclen x bsz x input_embed_dim
        x=self.dropout_input(input).expand(masked_source_hids.size(0),-1,-1)

        #Apply same dropout to all timesteps of source_hids, like in Nematus
        #mask = Bernoulli(torch.full_like(source_hids[0], 1 - self.dropout_p)).sample()/(1 - self.dropout_p)
        #source_hids=source_hids*mask # In theory, with broadcasting we multiply all timesteps by the same mask

        #Concatenate with source_hids
        x=torch.cat((x,masked_source_hids),-1)

        #Apply linear layer +  tanh
        x=self.activ_input_proj(self.input_proj(x))
        #x: srclen x bsz x alignment_dim

        #Reduce to one score
        attn_scores= torch.squeeze(self.score_proj(x),-1)
        #x: srclen x bsz

        #Apply softmax
        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        #Create context vector
        x = (attn_scores.unsqueeze(2) * masked_source_hids).sum(dim=0)

        return x, attn_scores

class ConditionalGru(nn.Module):
    def __init__(self,input_embed_dim, source_context_dim, hidden_dim,dropout=0.0):
        super().__init__()
        self.input_embed_dim=input_embed_dim
        self.source_context_dim=source_context_dim
        self.hidden_dim=hidden_dim

        self.dropout=dropout

        self.gru1=nn.GRUCell(input_size=self.input_embed_dim,hidden_size=self.hidden_dim)
        #TODO: configure dropout
        self.attention=ConcatAttentionLayer(input_embed_dim=self.hidden_dim, source_embed_dim=source_context_dim,alignment_dim=self.hidden_dim, bias=True, dropout=0.0)#bias = True like Bahdanau
        self.gru2=nn.GRUCell(input_size=self.source_context_dim, hidden_size=self.hidden_dim)

    def initialize_minibatch(self,encoder_outs):
        bsz=encoder_outs.size(1)
        #Mask for applying the same dropout to cand hidden states in all timesteps
        self.dropout_cand_hidden_mask= torch.ones([bsz ,self.hidden_dim ], dtype=encoder_outs.dtype, device=encoder_outs.device)
        self.dropout_cand_hidden_mask=F.dropout(self.dropout_cand_hidden_mask,p=self.dropout, training=self.training)

        self.precomputed_masked=self.attention.precompute_masked_source_hids(encoder_outs) if self.training else encoder_outs

        #Mask for applying the same dropout to new_hidden states in all timesteps
        #Not needed: it is applied outside
        #self.dropout_new_hidden_mask= torch.ones([bsz ,self.hidden_dim ], dtype=encoder_outs.dtype, device=encoder_outs.device)
        #self.dropout_new_hidden_mask=F.dropout(self.dropout_new_hidden_mask,p=self.dropout, training=self.training)

    def forward(self,prev_hidden,prev_emb,encoder_padding_mask):
        cand_hidden=self.gru1(prev_emb,prev_hidden)
        cand_hidden=cand_hidden*self.dropout_cand_hidden_mask
        #TODO: precompute dropout
        context,attn_scores=self.attention(cand_hidden,self.precomputed_masked,encoder_padding_mask)
        new_hidden=self.gru2(context,cand_hidden)
        return new_hidden,attn_scores,context

class GRUDecoder(FairseqIncrementalDecoder):
    """GRU decoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, cond_gru=False, adaptive_softmax_cutoff=None,debug=False
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True

        self.cond_gru=cond_gru

        self.debug=debug

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units

        #linear + tanh for initial state
        #TODO: we are assuming encoder is always bidirectional
        self.linear_initial_state=Linear(encoder_output_units,hidden_size,dropout=dropout_in)
        self.activ_initial_state=nn.Tanh()

        #TODO: should we apply droput here?
        if not self.cond_gru:
            self.layers = nn.ModuleList([
                # LSTM used custom initialization here
                nn.GRUCell(
                    input_size=encoder_output_units + embed_dim if layer == 0 else hidden_size,
                    hidden_size=hidden_size,
                )
                for layer in range(num_layers)
            ])
            if attention:
                # TODO make bias configurable
                self.attention = ConcatAttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=True, dropout=0.0)#bias = True like Bahdanau
            else:
                self.attention = None
        else:
            assert num_layers == 1
            self.attention=None
            self.layers = nn.ModuleList([
                # LSTM used custom initialization here
                ConditionalGru(
                    input_embed_dim=embed_dim,
                    source_context_dim=encoder_output_units,
                    hidden_dim=hidden_size,
                )
                for layer in range(num_layers)
            ])

        #Deep output
        self.logit_lstm=Linear(hidden_size, out_embed_dim, dropout=dropout_out)
        self.logit_prev=Linear(out_embed_dim, out_embed_dim, dropout=dropout_out)
        self.logit_ctx=Linear(encoder_output_units, out_embed_dim, dropout=dropout_out)
        self.activ_deep_output=nn.Tanh()

        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, embed_dim, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward(self, prev_output_tokens, encoder_out_dict, incremental_state=None):
        encoder_out = encoder_out_dict['encoder_out']
        encoder_padding_mask = encoder_out_dict['encoder_padding_mask']

        if self.debug:
            print("GRUDecoder forward")
            print("prev_output_tokens size: {} ".format(prev_output_tokens.size()))
            print("prev_output_tokens: {}".format(prev_output_tokens))
            print("encoder_padding_mask ({}): {}".format(encoder_padding_mask.size(),encoder_padding_mask))

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, encoder_hiddens = encoder_out[:2]
        srclen = encoder_outs.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        #Like in nematus, dropout all the embedding dimensions of a word at the
        #same time
        #
        # Nematus uses tf.layers.dropout(x, noise_shape=(tf.shape(x)[0], tf.shape(x)[1], 1)
        # See https://pytorch.org/docs/stable/nn.html#dropout-layers and https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400/2
        # Reverted to 1d
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        #bsz x seqlen x hidden_size
        logit_prev_input=x

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens = cached_state
        else:
            num_layers = len(self.layers)

            #Copy initialization from Nematus:
            # - Concatenate layers: IMPOSSIBLE, we only have access to last layer
            # - Average over time
            # - Apply FF + tanh

            #shape of encoder_outs: (seq_len,bsz,num_directions*hidden_size)
            # shape of encoder_padding_mask: (seq_len,bsz)
            # shape of division: (bsz,num_directions*hidden_size)/ (bsz): we add unsqueeze(1) to make dimensions match
            #print("encoder_outs: {}".format(encoder_outs))
            #print("encoder_padding_mask: {}".format(encoder_padding_mask))
            avg_states_num=torch.sum(encoder_outs,0)
            avg_states_denom=encoder_outs.size(0) -  torch.sum(encoder_padding_mask,0).unsqueeze(1).type_as(encoder_outs) if encoder_padding_mask is not None else encoder_outs.size(0)

            avg_states=torch.div( avg_states_num , avg_states_denom  )

            #print("avg_states_num({}): {}".format(avg_states_num.size(),avg_states_num))
            #print("avg_states_denom({}): {}".format(avg_states_denom.size() if not isinstance(avg_states_denom,int) else 0,avg_states_denom))
            #print("avg_states({}): {}".format(avg_states.size(),avg_states))

            #shape: (bsz,num_directions*hidden_size)
            hidden=self.activ_initial_state(self.linear_initial_state(avg_states))
            #shape: (bsz,decoder_hidden_size)

            #TODO: All layers have the same initial state: problematic!!!!
            prev_hiddens=[hidden for i in range(num_layers)]

        attn_scores = x.new_zeros(srclen, seqlen, bsz)
        outs = []
        context_vectors=[]

        prev_hiddens=list(prev_hiddens)
        if self.attention is not None:
            #Precompute masked source hidden states
            precomputed_masked=self.attention.precompute_masked_source_hids(encoder_outs) if self.training else encoder_outs


        if self.cond_gru:
            for i, rnn in enumerate(self.layers):
                rnn.initialize_minibatch(encoder_outs)

        for j in range(seqlen):
            # apply attention using the last layer's hidden state
            if self.attention is not None:
                context_vector, attn_scores[:, j, :] = self.attention(prev_hiddens[-1], precomputed_masked, encoder_padding_mask)
            else:
                context_vector = encoder_outs[0] #TODO: it should be the last state

            # input to GRU: concatenate context vector and input embeddings
            if self.cond_gru:
                input = x[j, :, :]
            else:
                input = torch.cat((x[j, :, :], context_vector), dim=1)

            #TODO: multi-layer IS WRONG
            assert len(self.layers) == 1
            for i, rnn in enumerate(self.layers):
                # recurrent cell:
                if self.cond_gru:
                    hidden,attn_scores[:, j, :],context_vector = rnn(prev_hiddens[i],input,encoder_padding_mask)
                else:
                    hidden = rnn(input, prev_hiddens[i])

                #Apply dropout to hidden
                hidden = F.dropout(hidden, p=self.dropout_out, training=self.training)
                # hidden state becomes the input to the next layer
                #input = F.dropout(hidden, p=self.dropout_out, training=self.training)
                #This is not needed anymore: input feeding is disabled in Bahdanau

                # save state for next time step
                prev_hiddens[i] = hidden

            context_vectors.append(context_vector)
            #Hidden state of top layer
            out=hidden

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            prev_hiddens,
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        #x: bsz x seqlen x hidden_size

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        #deep output like nematus
        logit_ctx_out=self.logit_ctx( torch.stack(context_vectors).transpose(0,1)  )
        logit_prev_out=self.logit_prev(logit_prev_input)
        logit_lstm_out=self.logit_lstm(x)

        x=self.activ_deep_output(logit_ctx_out + logit_prev_out + logit_lstm_out )

        # project back to size of vocabulary
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)

        #print("Forward pass.\nx({}):{}\nattn_scores:{}".format(x.size(),x,attn_scores))
        return x, attn_scores

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

class GRUDecoderTwoInputs(FairseqIncrementalDecoder):
    """GRU decoder."""
    def __init__(
        self, dictionary,dictionary_b, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None, pretrained_embed_b=None,
        share_input_output_embed=False, b_condition_end=False , cond_gru=False , adaptive_softmax_cutoff=None,debug=False
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True

        self.debug=debug

        self.b_condition_end = b_condition_end
        self.cond_gru=cond_gru

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        num_embeddings_b=len(dictionary_b) if dictionary_b else None
        padding_idx = dictionary.pad()
        padding_idx_b = dictionary_b.pad() if dictionary_b else None

        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        if pretrained_embed_b is None:
            if dictionary_b:
                #At the moment, both embeddings have the same size
                self.embed_tokens_b=Embedding(num_embeddings_b, embed_dim, padding_idx_b)
            else:
                self.embed_tokens_b=None
        else:
            self.embed_tokens_b=pretrained_embed_b

        self.encoder_output_units = encoder_output_units

        #linear + tanh for initial state
        #TODO: we are assuming encoder is always bidirectional
        self.linear_initial_state=Linear(encoder_output_units,hidden_size,dropout=dropout_in)
        self.activ_initial_state=nn.Tanh()

        if not self.cond_gru:
            self.layers = nn.ModuleList([
                # LSTM used custom initialization here
                nn.GRUCell(
                    input_size=( (encoder_output_units + embed_dim + (embed_dim if self.embed_tokens_b else hidden_size) ) if not self.b_condition_end else (encoder_output_units + embed_dim) ) if layer == 0 else hidden_size,
                    hidden_size=hidden_size,
                )
                for layer in range(num_layers)
            ])
            if attention:
                # TODO make bias configurable
                self.attention = ConcatAttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=True, dropout=0.0)#bias = True like Bahdanau
            else:
                self.attention = None
        else:
            self.attention=None
            self.layers = nn.ModuleList([
                ConditionalGru(
                    input_embed_dim=embed_dim+(embed_dim if self.embed_tokens_b else hidden_size) if not self.b_condition_end else embed_dim,
                    source_context_dim=encoder_output_units,
                    hidden_dim=hidden_size,
                )
                for layer in range(num_layers)
            ])

        #Deep output
        self.logit_lstm=Linear(hidden_size, out_embed_dim, dropout=dropout_out)
        self.logit_prev=Linear(embed_dim*2 if not self.b_condition_end else embed_dim, out_embed_dim, dropout=dropout_out)
        if self.b_condition_end:
            self.logit_tag = Linear((embed_dim if self.embed_tokens_b else hidden_size), out_embed_dim, dropout=dropout_out)
        self.logit_ctx=Linear(encoder_output_units, out_embed_dim, dropout=dropout_out)
        self.activ_deep_output=nn.Tanh()

        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, embed_dim, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward(self, prev_output_tokens,prev_output_tokens_b, encoder_out_dict, incremental_state=None):
        encoder_out = encoder_out_dict['encoder_out']
        encoder_padding_mask = encoder_out_dict['encoder_padding_mask']

        if self.debug:
            print("GRUDecoderTwoInputs forward")
            print("prev_output_tokens size: {} ".format(prev_output_tokens.size()))
            print("prev_output_tokens: {} ".format(prev_output_tokens))
            print("prev_output_tokens_b size: {} ".format(prev_output_tokens_b.size()))
            print("prev_output_tokens_b: {} ".format(prev_output_tokens_b))
            print("encoder_padding_mask: {}".format(encoder_padding_mask))

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            #TODO: careful with this at decoding time
            prev_output_tokens_b=prev_output_tokens_b[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, encoder_hiddens = encoder_out[:2]
        srclen = encoder_outs.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        #embed additional tokens
        if self.embed_tokens_b:
            x_b=self.embed_tokens_b(prev_output_tokens_b)
        else:
            #x_b represents a hidden state
            feedback_encoder_outs, feedback_encoder_outs = prev_output_tokens_b['encoder_out'][:2]
            #shape of feedback_encoder_outs: (seq_len,bsz,hidden_size)
            x_b=feedback_encoder_outs.transpose(0, 1)
        x_b = F.dropout(x_b, p=self.dropout_in, training=self.training)
        logit_tag_input=x_b

        #Concatenate both
        if not self.b_condition_end:
            x=torch.cat((x, x_b), dim=-1)

        #bsz x seqlen x hidden_size
        logit_prev_input=x

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens = cached_state
        else:
            num_layers = len(self.layers)

            #Copy initialization from Nematus:
            # - Concatenate layers: IMPOSSIBLE, we only have access to last layer
            # - Average over time
            # - Apply FF + tanh

            #shape of encoder_outs: (seq_len,bsz,num_directions*hidden_size)
            # shape of encoder_padding_mask: (seq_len,bsz)
            # shape of division: (bsz,num_directions*hidden_size)/ (bsz): we add unsqueeze(1) to make dimensions match
            #print("encoder_outs: {}".format(encoder_outs))
            #print("encoder_padding_mask: {}".format(encoder_padding_mask))
            avg_states_num=torch.sum(encoder_outs,0)
            avg_states_denom=encoder_outs.size(0) -  torch.sum(encoder_padding_mask,0).unsqueeze(1).type_as(encoder_outs) if encoder_padding_mask is not None else encoder_outs.size(0)

            avg_states=torch.div( avg_states_num , avg_states_denom  )

            #print("avg_states_num({}): {}".format(avg_states_num.size(),avg_states_num))
            #print("avg_states_denom({}): {}".format(avg_states_denom.size() if not isinstance(avg_states_denom,int) else 0,avg_states_denom))
            #print("avg_states({}): {}".format(avg_states.size(),avg_states))

            #shape: (bsz,num_directions*hidden_size)
            hidden=self.activ_initial_state(self.linear_initial_state(avg_states))
            #shape: (bsz,decoder_hidden_size)

            #TODO: All layers have the same initial state: problematic!!!!
            prev_hiddens=[hidden for i in range(num_layers)]

        attn_scores = x.new_zeros(srclen, seqlen, bsz)
        outs = []
        context_vectors=[]

        prev_hiddens=list(prev_hiddens)
        if self.attention is not None:
            #Precompute masked source hidden states
            precomputed_masked=self.attention.precompute_masked_source_hids(encoder_outs) if self.training else encoder_outs

        if self.cond_gru:
            for i, rnn in enumerate(self.layers):
                rnn.initialize_minibatch(encoder_outs)

        for j in range(seqlen):
            # apply attention using the last layer's hidden state
            if self.attention is not None:
                context_vector, attn_scores[:, j, :] = self.attention(prev_hiddens[-1], precomputed_masked, encoder_padding_mask)
            else:
                context_vector = encoder_outs[0] #TODO: it should be the last state

            # input to GRU: concatenate context vector and input embeddings
            if self.cond_gru:
                input = x[j, :, :]
            else:
                input = torch.cat((x[j, :, :], context_vector), dim=1)

            assert len(self.layers) == 1
            for i, rnn in enumerate(self.layers):
                if self.cond_gru:
                    hidden,attn_scores[:, j, :],context_vector = rnn(prev_hiddens[i],input,encoder_padding_mask)
                else:
                    hidden = rnn(input, prev_hiddens[i])

                hidden = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden

            context_vectors.append(context_vector)

            #Hidden state of top layer
            out=hidden

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            prev_hiddens,
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        #x: bsz x seqlen x hidden_size

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        #deep output like nematus
        logit_ctx_out=self.logit_ctx( torch.stack(context_vectors).transpose(0,1)  )
        logit_prev_out=self.logit_prev(logit_prev_input)
        if self.b_condition_end:
            logit_tag_out=self.logit_tag(logit_tag_input)
        logit_lstm_out=self.logit_lstm(x)

        if self.b_condition_end:
            x=self.activ_deep_output(logit_ctx_out + logit_prev_out + logit_lstm_out + logit_tag_out)
        else:
            x=self.activ_deep_output(logit_ctx_out + logit_prev_out + logit_lstm_out )

        # project back to size of vocabulary
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)

        #print("Forward pass.\nx({}):{}\nattn_scores:{}".format(x.size(),x,attn_scores))
        return x, attn_scores

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


class Linear(nn.Module):
    def __init__(self,in_features, out_features, bias=True, dropout=0):
        super(Linear, self).__init__()
        self.layer=nn.Linear(in_features, out_features, bias=bias)
        self.layer.weight.data.uniform_(-0.1, 0.1)
        if bias:
            self.layer.bias.data.uniform_(-0.1, 0.1)
        self.dropout=nn.Dropout(dropout)
        self.dropout_short=nn.Dropout(dropout)
    def forward(self,x):
        x=self.layer(x)
        if len(x.size()) < 3:
            x=self.dropout_short(x)
        else:
            #in: bsz x seq_len x channel
            x=self.dropout(x)
        return x


#def Linear(in_features, out_features, bias=True, dropout=0):
#    """Linear layer (input: N x T x C)"""
#    m = nn.Linear(in_features, out_features, bias=bias)
#    m.weight.data.uniform_(-0.1, 0.1)
#    if bias:
#        m.bias.data.uniform_(-0.1, 0.1)
#    d=nn.Dropout(dropout)
#    return nn.Sequential(m,d)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

@register_model_architecture('bahdanau_rnn_two_decoders_sync', 'bahdanau_rnn_two_decoders_sync')
def bahdanau_rnn(args):
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 1024)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 1024)
    args.encoder_bidirectional= getattr(args, 'encoder_bidirectional', True)


@register_model_architecture('bahdanau_rnn_two_encdecoders_sync', 'bahdanau_rnn_two_encdecoders_sync')
def bahdanau_rnn(args):
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 1024)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 1024)
    args.encoder_bidirectional= getattr(args, 'encoder_bidirectional', True)


@register_model_architecture('bahdanau_rnn_two_encdecoders_async', 'bahdanau_rnn_two_encdecoders_async')
def bahdanau_rnn(args):
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 1024)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 1024)
    args.encoder_bidirectional= getattr(args, 'encoder_bidirectional', True)

@register_model_architecture('bahdanau_rnn_two_decoders_async', 'bahdanau_rnn_two_decoders_async')
def bahdanau_rnn(args):
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 1024)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 1024)
    args.encoder_bidirectional= getattr(args, 'encoder_bidirectional', True)

@register_model_architecture('bahdanau_rnn_two_decoders_mutual_influence_async', 'bahdanau_rnn_two_decoders_mutual_influence_async')
def bahdanau_rnn(args):
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 1024)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 1024)
    args.encoder_bidirectional= getattr(args, 'encoder_bidirectional', True)


@register_model_architecture('bahdanau_rnn', 'bahdanau_rnn')
def bahdanau_rnn(args):
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 1024)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 1024)
    args.encoder_bidirectional= getattr(args, 'encoder_bidirectional', True)
