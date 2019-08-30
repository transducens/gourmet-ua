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
        if args.decoder_freeze_embed and pretrained_decoder_embed != None:
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
    def __init__(self, encoder, decoder, decoder_b,encoder_b):
        BaseFairseqModel.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_b = decoder_b
        self.encoder_b=encoder_b
        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, GRUDecoderTwoInputs)
        assert self.decoder_b is None or isinstance(self.decoder_b, GRUDecoder) or isinstance(self.decoder_b, GRUDecoderTwoInputs)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        #Hack to call parent staticmethod
        BahdanauRNNModel.add_args(parser)
        parser.add_argument('--tags-condition-end', default=False, action='store_true',
                            help='Tags condition surface form decoder only at the end, as in lexical model. Surface forms condition tags decoder only at the end, as in lexical model.')
        parser.add_argument('--tags-condition-end-a', default=False, action='store_true',
                            help='Tags condition surface form decoder only at the end, as in lexical model.')
        parser.add_argument('--tags-condition-end-b', default=False, action='store_true',
                            help='Surface forms condition tags decoder only at the end, as in lexical model.')
        parser.add_argument('--gate-output-a', default=False, action='store_true',
                            help='Output of surface form decoder constains a gate to control influence of tags.')
        parser.add_argument('--surface-condition-tags', default=False, action='store_true',
                            help='Tag decoder has two inputs: previous timestep tag and previous timestep surface form')
        parser.add_argument('--decoder-b-ignores-encoder', default=False, action='store_true',
                            help='Auxiliary decoder ignores encoder')
        parser.add_argument('--two-encoders', default=False, action='store_true',
                            help='One encoder for each output')
        parser.add_argument('--encoders-share-embeddings', default=False, action='store_true',
                            help='If there are two encoders, they share SL word embeddings')
        parser.add_argument('--decoders-share-state-attention', default=False, action='store_true',
                            help='The two decoders share state and attention, but time shifted.')
        parser.add_argument('--decoders-share-state-attention-logits', default=False, action='store_true',
                            help='The two decoders share state,attention and deep output, but time shifted.')
        parser.add_argument('--share-embeddings-two-decoders', default=False, action='store_true',
                            help='Both decoders share embeddings')
        parser.add_argument('--share-factors-embeddings-two-decoders', default=False, action='store_true',
                            help='Both decoders share ONLY factors embeddings')
        parser.add_argument('--freeze-encoder-weights', default=False, action='store_true',
                            help='Freeze encoder weights')
        parser.add_argument('--decoder-freeze-factor-embed', default=False, action='store_true',
                            help='Freeze factor embeddings')
        parser.add_argument('--decoder-a-freeze-not-logits', default=False, action='store_true',
                            help='Freeze surface form decoder, do not freeze logits.')
        parser.add_argument('--dropout-conditioning-tags', type=float,default=0.0,
                            help='Dropout full conditioning tags.')



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
            pretrained_encoder_embed = pretrained_encoder_embed_b = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )
            if getattr(args,'two_encoders',False):
                if getattr(args,'encoders_share_embeddings',False):
                    pretrained_encoder_embed_b=pretrained_encoder_embed
                else:
                    pretrained_encoder_embed_b = Embedding(
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

            if args.share_embeddings_two_decoders or args.share_factors_embeddings_two_decoders:
                if pretrained_decoder_embed is None:
                    #Otherwise, pretrained_decoder_embed will be None and will be independently learnt by each decoder
                    if args.share_embeddings_two_decoders:
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
        if args.decoder_freeze_embed and pretrained_decoder_embed != None:
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

        if getattr(args,'two_encoders',False):
            if args.encoder_freeze_embed:
                pretrained_encoder_embed_b.weight.requires_grad = False
            encoder_b=GRUEncoder(
                dictionary=task.source_dictionary,
                embed_dim=args.encoder_embed_dim,
                hidden_size=args.encoder_hidden_size,
                num_layers=args.encoder_layers,
                dropout_in=args.encoder_dropout_in,
                dropout_out=args.encoder_dropout_out,
                bidirectional=args.encoder_bidirectional,
                pretrained_embed=pretrained_encoder_embed_b,
                debug=args.debug if 'debug' in args else False
            )
        else:
            encoder_b=None

        if 'freeze_encoder_weights' in args and  args.freeze_encoder_weights:
            encoder.freeze_weights()

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
            b_condition_end=args.tags_condition_end or getattr(args,'tags_condition_end_a',None),
            cond_gru=args.cond_gru if 'cond_gru' in args else False,
            gate_combination= getattr(args,'gate_output_a',False),
            two_outputs=getattr(args,'decoders_share_state_attention',False) or getattr(args,'decoders_share_state_attention_logits',False),
            two_outputs_share_logits=getattr(args,'decoders_share_state_attention_logits',False),
            dropout_cond_tags=getattr(args,'dropout_conditioning_tags',0.0),
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            debug=args.debug if 'debug' in args else False
        )

        decoder_b=None
        if not ( getattr(args,'decoders_share_state_attention',False) or getattr(args,'decoders_share_state_attention_logits',False)):
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
                    attention=options.eval_bool(args.decoder_attention) and not getattr(args,'decoder_b_ignores_encoder',False),
                    encoder_output_units=encoder.output_units,
                    pretrained_embed=pretrained_decoder_embed_b,
                    pretrained_embed_b=pretrained_decoder_embed,
                    share_input_output_embed=args.share_decoder_input_output_embed,
                    b_condition_end=args.tags_condition_end or getattr(args,'tags_condition_end_b',None),
                    cond_gru=getattr(args,'cond_gru',False) and not getattr(args,'decoder_b_ignores_encoder',False),
                    ignore_encoder_input=getattr(args,'decoder_b_ignores_encoder',False),
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

        if  getattr(args,'decoder_a_freeze_not_logits',None):
            decoder.freeze_weights(freeze_logits=False)

        if  getattr(args,'decoder_freeze_embed',None):
            decoder.embed_tokens.weight.requires_grad=False
            if args.surface_condition_tags and decoder_b is not None:
                decoder_b.embed_tokens_b.weight.requires_grad=False

        if  getattr(args,'decoder_freeze_factor_embed',None):
            decoder.embed_tokens_b.weight.requires_grad=False
            if decoder_b is not None:
                decoder_b.embed_tokens.weight.requires_grad=False

        r= cls(encoder, decoder, decoder_b,encoder_b)
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
        encoder_b_out=None
        if self.encoder_b != None:
            encoder_b_out = self.encoder_b(src_tokens, src_lengths)
        else:
            encoder_b_out=encoder_out

        if self.decoder.two_outputs:
            input_1= prev_output_tokens.new_zeros([prev_output_tokens.size(0),prev_output_tokens.size(1)*2])
            input_2= prev_output_tokens.new_zeros([prev_output_tokens.size(0),prev_output_tokens.size(1)*2])
            for i in range(prev_output_tokens.size(1)):
                input_1[:,2*i+1]=prev_output_tokens[:,i]
                input_2[:,2*i+1]=cur_output_factors[:,i]
            for i in range(prev_output_factors.size(1)):
                input_1[:,2*i]=prev_output_tokens[:,i]
                input_2[:,2*i]=prev_output_factors[:,i]
            decoder_out, decoder_b_out = self.decoder(input_1,input_2, encoder_out)
            return decoder_out, decoder_b_out
        else:
            decoder_out = self.decoder(prev_output_tokens,cur_output_factors, encoder_out)
            if isinstance(self.decoder_b,GRUDecoderTwoInputs):
                decoder_b_out = self.decoder_b(prev_output_factors,prev_output_tokens, encoder_b_out)
            else:
                decoder_b_out = self.decoder_b(prev_output_factors, encoder_b_out)
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
        if args.decoder_freeze_embed and pretrained_decoder_embed != None:
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
        if args.decoder_freeze_embed and pretrained_decoder_embed != None:
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
    def __init__(self, encoder, decoder, decoder_b, feedback_encoder,feedback_state_and_last_subword_embs,apply_transformation_input_b,reset_b_decoder=False):
        BaseFairseqModel.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_b = decoder_b
        self.feedback_encoder= feedback_encoder
        self.feedback_state_and_last_subword_embs=feedback_state_and_last_subword_embs
        self.apply_transformation_input_b=apply_transformation_input_b

        if self.apply_transformation_input_b:
            self.linear_transf_input_b=Linear(self.decoder.hidden_size, self.decoder.hidden_size, bias=True)
            self.activ_transf_input_b=nn.Tanh()
        else:
            self.linear_transf_input_b=None
            self.activ_transf_input_b=None

        self.reset_b_decoder=reset_b_decoder

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
        parser.add_argument('--tags-condition-end-a', default=False, action='store_true',
                            help='Tags condition surface form decoder only at the end, as in lexical model')
        parser.add_argument('--tags-condition-end-b', default=False, action='store_true',
                            help='Tags condition surface form decoder only at the end, as in lexical model')
        parser.add_argument('--feedback-encoder', default=False, action='store_true',
                            help='Use an encoder to condense all the previous surface forms.')
        parser.add_argument('--feedback-state-and-last-subword', default=False, action='store_true',
                            help='Use concatenation of decoder hidden state and last subword for feedback to tags decoder.')
        parser.add_argument('--transform-last-state', default=False, action='store_true',
                            help='When using concatenation of decoder hidden state and last subword for feedback to tags decoder, apply MLP to transfom state.')
        parser.add_argument('--share-embeddings-two-decoders', default=False, action='store_true',
                            help='Both decoders share embeddings')
        parser.add_argument('--share-factors-embeddings-two-decoders', default=False, action='store_true',
                            help='Both decoders share ONLY factors embeddings')
        parser.add_argument('--freeze-encoder-weights', default=False, action='store_true',
                            help='Freeze encoder weights')
        parser.add_argument('--decoder-freeze-factor-embed', default=False, action='store_true',
                            help='Freeze factor embeddings')
        parser.add_argument('--decoder-a-freeze', default=False, action='store_true',
                            help='Compleely freeze surface form decoder')
        parser.add_argument('--decoder-b-freeze', default=False, action='store_true',
                            help='Compleely freeze factor decoder')
        parser.add_argument('--reset-b-decoder', default=False, action='store_true',
                            help='Reset factors decoder when loading a model')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance.
        Encoders and decoders now are GRUs. Initialization of hidden state
        and output layers similar to Nematus
         """

        #import pdb; pdb.set_trace()

        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if 'feedback_state_and_last_subword' not in args:
            args.feedback_state_and_last_subword=None

        #Now we manage embedding creation
        #if args.feedback_state_and_last_subword and not args.share_embeddings_two_decoders:
        #    raise ValueError('--feedback_state_and_last_subword must match --share_embeddings_two_decoders')

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
            if args.share_embeddings_two_decoders or args.share_factors_embeddings_two_decoders:
                if pretrained_decoder_embed is None:
                    #Otherwise, pretrained_decoder_embed will be None and will be independently learnt by each decoder
                    if args.share_embeddings_two_decoders:
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
        if args.decoder_freeze_embed and pretrained_decoder_embed != None:
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
        if getattr(args,'freeze_encoder_weights',None):
            encoder.freeze_weights()

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
            if getattr(args,'freeze_encoder_weights',None):
                feedback_encoder.freeze_weights()

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
            b_condition_end=args.tags_condition_end or getattr(args,'tags_condition_end_a',None),
            cond_gru=args.cond_gru if 'cond_gru' in args else False,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            debug=args.debug if 'debug' in args else False
        )
        if 'decoder_a_freeze' in args and args.decoder_a_freeze:
            decoder.freeze_weights()

        size_input_b=None
        if args.feedback_encoder:
            size_input_b=feedback_encoder.hidden_size
        elif args.feedback_state_and_last_subword:
            size_input_b=(decoder.hidden_size + args.decoder_embed_dim)
        decoder_b = GRUDecoderTwoInputs(
            dictionary=task.target_factors_dictionary,
            dictionary_b=task.target_dictionary if (not args.feedback_encoder and not args.feedback_state_and_last_subword ) else None, # empty dictionary means that we do not need to apply embeddings to second output
            size_input_b=size_input_b,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed_b,
            pretrained_embed_b=pretrained_decoder_embed if (not args.feedback_encoder and not args.feedback_state_and_last_subword ) else None,
            share_input_output_embed=args.share_decoder_input_output_embed,
            cond_gru=args.cond_gru if 'cond_gru' in args else False,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            debug=args.debug if 'debug' in args else False
        )
        if 'decoder_b_freeze' in args and  args.decoder_b_freeze:
            decoder_b.freeze_weights()
        feedback_state_and_last_subword_embs=None
        if args.feedback_state_and_last_subword:
            feedback_state_and_last_subword_embs=pretrained_decoder_embed
            if feedback_state_and_last_subword_embs == None:
                #If we are not sharing surface form embeddings between the two decoders, create them
                feedback_state_and_last_subword_embs=Embedding(len(task.target_dictionary), args.decoder_embed_dim, task.target_dictionary.pad())
            if 'decoder_b_freeze' in args and args.decoder_b_freeze:
                feedback_state_and_last_subword_embs.weight.requires_grad=False

        #Properly freeze embeddings according to args
        if args.decoder_freeze_embed:
            decoder.embed_tokens.weight.requires_grad = False
            if decoder_b.embed_tokens_b is not None:
                decoder_b.embed_tokens_b.weight.requires_grad=False
        if getattr(args,'decoder_freeze_factor_embed',None):
            decoder.embed_tokens_b.weight.requires_grad = False
            decoder_b.embed_tokens.weight.requires_grad = False

        r= cls(encoder, decoder, decoder_b, feedback_encoder,feedback_state_and_last_subword_embs=feedback_state_and_last_subword_embs,apply_transformation_input_b=args.transform_last_state, reset_b_decoder=args.reset_b_decoder if 'reset_b_decoder' in args else False)
        return r

    def load_state_dict(self,state_dict, strict=True):
        #import pdb; pdb.set_trace()
        if self.reset_b_decoder:
            #TODO: additional encoder is not reset
            reset_keys=[k for k in state_dict.keys() if k.startswith("decoder_b.") or k.startswith("feedback_state_and_last_subword_embs.") or k.startswith("linear_transf_input_b.")]

            #If surface form embeddings are shared, we do not reset them
            if self.decoder.embed_tokens == self.decoder_b.embed_tokens_b or self.decoder.embed_tokens == self.feedback_state_and_last_subword_embs:
                reset_keys= [k for k in reset_keys if not k.startswith("decoder_b.embed_tokens_b") and not k.startswith("feedback_state_and_last_subword_embs") ]

            #If factor embeddings are shared, we do not reset them
            if self.decoder_b.embed_tokens == self.decoder.embed_tokens_b:
                reset_keys= [k for k in reset_keys if not k.startswith("decoder_b.embed_tokens")]

            for k in reset_keys:
                state_dict.pop(k)

            print("Keys loaded: {}".format(state_dict.keys()))
            print("Keys reset: {}".format(reset_keys))

            #We set strict to False because we removed some keys
            super().load_state_dict(state_dict, False)
        else:
            super().load_state_dict(state_dict, strict)

    def get_target_factors(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target_factors_async']

    def forward(self, src_tokens, src_lengths, prev_output_tokens, prev_output_factors, cur_output_factors, prev_output_tokens_lengths,prev_output_tokens_word_end_positions,prev_output_tokens_last_subword=None,prev_output_tokens_first_subword=None ):
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
        #print("Forward: prev_output_tokens:{}\nprev_output_factors:{}\ncur_output_factors:{}\nprev_output_tokens_last_subword:{}\n".format(prev_output_tokens, prev_output_factors, cur_output_factors,prev_output_tokens_last_subword))


        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens,cur_output_factors, encoder_out)
        if self.feedback_encoder is not None:
            feedback_encoder_out=self.feedback_encoder(prev_output_tokens, prev_output_tokens_lengths)
            feedback_encoder_outs, feedback_encoder_hiddens = feedback_encoder_out['encoder_out'][:2]
            #shape of feedback_encoder_outs=prev_output_tokens_b: (seq_len,bsz,hidden_size)
            feedback_encoder_outs=feedback_encoder_outs.transpose(0, 1)
            #now (bsz,seq_len,hidden_size)

            #print("Creating input for decoder B from additional RNN")
            #print("prev_output_tokens_word_end_positions: {}".format(prev_output_tokens_word_end_positions))
            #print("prev_output_tokens_word_end_positions lengths: {}".format([len(l) for l in prev_output_tokens_word_end_positions]))
            #print("feedback_encoder_outs {}: {}".format(feedback_encoder_outs.size(),feedback_encoder_outs))

            #Create a blank tensor and fill it with index-selected positions
            #New input has same size as prev_output_factors
            second_input_decoder_b= feedback_encoder_outs.new_zeros([feedback_encoder_outs.size(0),prev_output_factors.size(1),feedback_encoder_outs.size(2)])
            for batch_idx in range(len(prev_output_tokens_word_end_positions)):
                #second_input_decoder_b[batch_idx].index_copy_(0,prev_output_tokens_word_end_positions[batch_idx],feedback_encoder_outs[batch_idx])
                for seq_pos_idx,original_pos in enumerate(prev_output_tokens_word_end_positions[batch_idx]):
                    second_input_decoder_b[batch_idx,seq_pos_idx,:]=feedback_encoder_outs[batch_idx,original_pos,:]
        elif self.feedback_state_and_last_subword_embs is not None:
            #( seq_len,bsz,hidden_size )
            all_hiddens_last_layer=torch.stack(decoder_out[2])
            #( bsz,seq_len+1,hidden_size )
            all_hiddens_last_layer=all_hiddens_last_layer.transpose(0,1)

            if self.apply_transformation_input_b:
                all_hiddens_last_layer=self.linear_transf_input_b(all_hiddens_last_layer)
                all_hiddens_last_layer=self.activ_transf_input_b(all_hiddens_last_layer)

            second_input_decoder_b_a= decoder_out[0].new_zeros([all_hiddens_last_layer.size(0),prev_output_factors.size(1),all_hiddens_last_layer.size(2)])
            for batch_idx in range(len(prev_output_tokens_word_end_positions)):
                for seq_pos_idx,original_pos in enumerate(prev_output_tokens_word_end_positions[batch_idx]):
                    second_input_decoder_b_a[batch_idx,seq_pos_idx,:]=all_hiddens_last_layer[batch_idx,original_pos,:]

            second_input_decoder_b_b=self.feedback_state_and_last_subword_embs(prev_output_tokens_last_subword)

            second_input_decoder_b=torch.cat((second_input_decoder_b_a,second_input_decoder_b_b),-1)
        else:
            second_input_decoder_b=prev_output_tokens_first_subword
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
        if args.decoder_freeze_embed and pretrained_decoder_embed != None:
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

    def freeze_weights(self):
        #Freeze embeddings
        self.embed_tokens.weight.requires_grad=False

        #Freeze GRU
        for att in [ 'weight_ih_l', 'weight_hh_l','bias_ih_l' , 'bias_hh_l' ]:
            for i in range(self.rnn.num_layers):
                getattr(self.rnn,att+str(i)).requires_grad=False
                if self.rnn.bidirectional:
                    getattr(self.rnn,att+str(i)+"_reverse").requires_grad=False


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

    def freeze_weights(self):
        self.input_proj.freeze_weights()
        self.score_proj.freeze_weights()

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

    def freeze_weights(self):
        self.attention.freeze_weights()
        for gru in [self.gru1, self.gru2]:
            for att in [ 'weight_ih', 'weight_hh','bias_ih' , 'bias_hh' ]:
                getattr(gru,att).requires_grad=False

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
        self, dictionary,dictionary_b,size_input_b=None, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None, pretrained_embed_b=None,
        share_input_output_embed=False, b_condition_end=False , cond_gru=False ,
        ignore_encoder_input=False , gate_combination=False, two_outputs=False, two_outputs_share_logits=False
        dropout_cond_tags=0.0, adaptive_softmax_cutoff=None,debug=False
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.dropout_cond_tags=dropout_cond_tags
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True

        self.ignore_encoder_input=ignore_encoder_input
        self.debug=debug

        self.b_condition_end = b_condition_end
        self.cond_gru=cond_gru
        self.gate_combination=gate_combination

        self.two_outputs=two_outputs
        self.two_outputs_share_logits=two_outputs_share_logits

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

        if self.ignore_encoder_input:
            encoder_output_units=0
            self.linear_initial_state=None
            self.activ_initial_state=None
        else:
            #linear + tanh for initial state
            #TODO: we are assuming encoder is always bidirectional
            self.linear_initial_state=Linear(encoder_output_units,hidden_size,dropout=dropout_in)
            self.activ_initial_state=nn.Tanh()

        assert not (self.ignore_encoder_input and self.cond_gru)

        if not self.cond_gru:
            self.layers = nn.ModuleList([
                # LSTM used custom initialization here
                nn.GRUCell(
                    input_size=( (encoder_output_units + embed_dim + (embed_dim if self.embed_tokens_b else size_input_b) ) if not self.b_condition_end else (encoder_output_units + embed_dim) ) if layer == 0 else hidden_size,
                    hidden_size=hidden_size,
                )
                for layer in range(num_layers)
            ])

            assert not (self.ignore_encoder_input and (attention == True))
            if attention:
                # TODO make bias configurable
                self.attention = ConcatAttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=True, dropout=0.0)#bias = True like Bahdanau
            else:
                self.attention = None
        else:
            self.attention=None
            self.layers = nn.ModuleList([
                ConditionalGru(
                    input_embed_dim=embed_dim+(embed_dim if self.embed_tokens_b else size_input_b) if not self.b_condition_end else embed_dim,
                    source_context_dim=encoder_output_units,
                    hidden_dim=hidden_size,
                )
                for layer in range(num_layers)
            ])

        if self.gate_combination:
            self.gate_linear_ctx=Linear(encoder_output_units, out_embed_dim, dropout=dropout_out)
            self.gate_linear_lstm=Linear(hidden_size, out_embed_dim, dropout=dropout_out)
            self.gate_activation=nn.Sigmoid()
            self.logit_prev_a=Linear(embed_dim, out_embed_dim, dropout=dropout_out)
            self.logit_prev_b=Linear(embed_dim if self.embed_tokens_b else size_input_b, out_embed_dim, dropout=dropout_out)
        else:
            self.logit_prev=Linear(embed_dim+(embed_dim if self.embed_tokens_b else size_input_b) if not self.b_condition_end else embed_dim, out_embed_dim, dropout=dropout_out)
            self.logit_prev_b=None

        #Deep output
        self.logit_lstm=Linear(hidden_size, out_embed_dim, dropout=dropout_out)
        if self.b_condition_end:
            self.logit_tag = Linear((embed_dim if self.embed_tokens_b else size_input_b), out_embed_dim, dropout=dropout_out)
        if self.ignore_encoder_input:
            self.logit_ctx=None
        else:
            self.logit_ctx=Linear(encoder_output_units, out_embed_dim, dropout=dropout_out)
        self.activ_deep_output=nn.Tanh()

        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, embed_dim, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

        if self.two_outputs:
            if self.gate_combination:
                assert False
            if not self.two_outputs_share_logits:
                self.logit_prev_b=Linear(embed_dim+(embed_dim if self.embed_tokens_b else size_input_b) if not self.b_condition_end else embed_dim, out_embed_dim, dropout=dropout_out)
                self.logit_lstm_b=Linear(hidden_size, out_embed_dim, dropout=dropout_out)
                if self.b_condition_end:
                    assert False
                if self.ignore_encoder_input:
                    self.logit_ctx_b=None
                else:
                    self.logit_ctx_b=Linear(encoder_output_units, out_embed_dim, dropout=dropout_out)
                self.activ_deep_output_b=nn.Tanh()

            self.fc_out_b =None
            self.adaptive_softmax_b=None
            if adaptive_softmax_cutoff is not None:
                # setting adaptive_softmax dropout to dropout_out for now but can be redefined
                self.adaptive_softmax_b = AdaptiveSoftmax(num_embeddings_b, embed_dim, adaptive_softmax_cutoff,
                                                        dropout=dropout_out)
            elif not self.share_input_output_embed:
                self.fc_out_b = Linear(out_embed_dim, num_embeddings_b, dropout=dropout_out)


    def freeze_weights(self, freeze_logits=True):
        #Freeze embeddings
        if self.embed_tokens:
            self.embed_tokens.weight.requires_grad=False
        if self.embed_tokens_b:
            self.embed_tokens_b.weight.requires_grad=False

        #Freeze linear initial state
        self.linear_initial_state.freeze_weights()

        if not self.cond_gru:
            #Freeze GRUCell and ConcatAttentionLayer
            for l in self.layers:
                for att in [ 'weight_ih', 'weight_hh','bias_ih' , 'bias_hh' ]:
                    getattr(l,att).requires_grad=False

            if self.attention:
                self.attention.freeze_weights()

        else:
            #Freeze ConditionalGrus
            for l in self.layers:
                l.freeze_weights()

        if freeze_logits:
            self.logit_lstm.freeze_weights()
            self.logit_prev.freeze_weights()
            if self.b_condition_end:
                self.logit_tag.freeze_weights()
            self.logit_ctx.freeze_weights()

            if self.two_outputs:
                assert self.b_condition_end == False
                if not self.two_outputs_share_logits:
                    self.logit_lstm_b.freeze_weights()
                    self.logit_ctx_b.freeze_weights()
                    self.logit_prev_b.freeze_weights()

                if  self.fc_out_b is not None:
                    self.fc_out_b.freeze_weights()
                if self.adaptive_softmax_b is not None:
                    self.adaptive_softmax_b.freeze_weights()

            assert self.adaptive_softmax == None

            if not self.share_input_output_embed:
                self.fc_out.freeze_weights()

    #if self.two_outputs, prev_output_tokens and prev_output_tokens_b are already interleaved
    def forward(self, prev_output_tokens,prev_output_tokens_b, encoder_out_dict, incremental_state=None,two_outputs_and_tag_generation=False):
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
            x_b=prev_output_tokens_b
            #x_b represents a hidden state
        x_b= F.dropout2d(x_b, p=self.dropout_cond_tags, training=self.training)
        x_b = F.dropout(x_b, p=self.dropout_in, training=self.training)
        logit_tag_input=x_b

        logit_prev_input_a=x
        logit_prev_input_b=x_b

        #Concatenate both
        if not self.b_condition_end:
            x=torch.cat((x, x_b), dim=-1)

        #bsz x seqlen x hidden_size
        logit_prev_input=x

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        all_hiddens_last_layer=[]

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens = cached_state
        else:
            num_layers = len(self.layers)

            if self.ignore_encoder_input:
                hidden=x.new_zeros(bsz,self.hidden_size)
            else:
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

        all_hiddens_last_layer.append(prev_hiddens[-1])
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
                if self.ignore_encoder_input:
                    input = x[j, :, :]
                else:
                    input = torch.cat((x[j, :, :], context_vector), dim=1)

            #TODO: multi-layer IS WRONG
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
            all_hiddens_last_layer.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            prev_hiddens,
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        if self.two_outputs:
            #TODO: if we are decoding, only one of x and x_b will be not None
            if incremental_state is not None or two_outputs_and_tag_generation:
                if two_outputs_and_tag_generation:
                    #We are generating a tag: use _b part
                    x_b=x
                    x=None
                else:
                    #We are generating a surface form: do not use _b part
                    x_b=None
            else:
                #Extract even and odd positions
                x_b=x[0::2]
                x=x[1::2]

        # T x B x C -> B x T x C
        if x is not None:
            x = x.transpose(1, 0)
            #x: bsz x seqlen x hidden_size

        if self.two_outputs and x_b is not None:
            x_b=x_b.transpose(1,0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        if x is not None:
            startingIndex=1
            #during decoding, starting index is 0, because we process only one symbol
            if incremental_state is not None:
                startingIndex=0

            #deep output like nematus
            if self.gate_combination:
                e=self.gate_activation(self.gate_linear_ctx(torch.stack(context_vectors).transpose(0,1))+self.gate_linear_lstm(x))
                logit_prev_out=e*self.logit_prev_a(logit_prev_input_a) + (1-e)*self.logit_prev_b(logit_prev_input_b)
            else:
                logit_prev_out=self.logit_prev(logit_prev_input) if not self.two_outputs else self.logit_prev(logit_prev_input[:,startingIndex::2])
            if self.b_condition_end:
                logit_tag_out=self.logit_tag(logit_tag_input)
            logit_lstm_out=self.logit_lstm(x)
            if not self.ignore_encoder_input:
                logit_ctx_out=self.logit_ctx( torch.stack(context_vectors).transpose(0,1)  ) if not self.two_outputs else self.logit_ctx( torch.stack(context_vectors[startingIndex::2]).transpose(0,1)  )
            else:
                logit_ctx_out=torch.zeros_like(logit_prev_out)

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

        if self.two_outputs and x_b is not None:
            logit_prev_out=self.logit_prev_b(logit_prev_input[:,0::2]) if not self.two_outputs_share_logits else self.logit_prev(logit_prev_input[:,0::2])
            logit_lstm_out=self.logit_lstm_b(x_b) if not self.two_outputs_share_logits else self.logit_lstm(x_b)
            if not self.ignore_encoder_input:
                logit_ctx_out=self.logit_ctx_b( torch.stack(context_vectors[0::2]).transpose(0,1)  ) if not self.two_outputs_share_logits else self.logit_ctx( torch.stack(context_vectors[0::2]).transpose(0,1)  )
            else:
                logit_ctx_out=torch.zeros_like(logit_prev_out)

            x_b=self.activ_deep_output_b(logit_ctx_out + logit_prev_out + logit_lstm_out ) if not self.two_outputs_share_logits else self.activ_deep_output(logit_ctx_out + logit_prev_out + logit_lstm_out )

            # project back to size of vocabulary
            if self.adaptive_softmax is None:
                if self.share_input_output_embed:
                    x_b = F.linear(x_b, self.embed_tokens_b.weight)
                else:
                    x_b = self.fc_out_b(x_b)

        if self.debug:
            print("all_hiddens_last_layer: {}".format(all_hiddens_last_layer))
        #print("Forward pass.\nx({}):{}\nattn_scores:{}".format(x.size(),x,attn_scores))

        if self.two_outputs:
            return (x, attn_scores,all_hiddens_last_layer),(x_b, attn_scores,all_hiddens_last_layer)
        else:
            return x, attn_scores,all_hiddens_last_layer

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
    def freeze_weights(self):
        self.layer.weight.requires_grad=False
        if self.layer.bias is not None:
            self.layer.bias.requires_grad=False

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
