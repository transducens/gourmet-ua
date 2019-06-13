import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.modules import AdaptiveSoftmax
from fairseq.models import (
    FairseqEncoder, FairseqIncrementalDecoder, BaseFairseqModel, register_model,
    register_model_architecture,
)
from fairseq.models.lstm import LSTMModel,LSTMEncoder,LSTMDecoder,base_architecture,Embedding,Linear,LSTMCell,AttentionLayer

from . import lstm_two_decoders_model

@register_model('lstm_two_decoders_async')
class LSTMTwoDecodersAsyncModel(LSTMModel):
    """
    add_args is completely inherited
    """
    def __init__(self, encoder, decoder, decoder_b):
        BaseFairseqModel.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_b = decoder_b
        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, lstm_two_decoders_model.LSTMDecoderTwoInputs)
        assert isinstance(self.decoder_b, LSTMDecoder)


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
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

        encoder = LSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
        )
        decoder = lstm_two_decoders_model.LSTMDecoderTwoInputs(
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
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
        )
        decoder_b=LSTMDecoder(
            dictionary=task.target_factors_dictionary,
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
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
        )
        return cls(encoder, decoder, decoder_b)

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
        #print("prev_output_tokens:({}){}\nprev_output_factors:({}){}\ncur_output_factors:({}){}".format(prev_output_tokens.size(),prev_output_tokens, prev_output_factors.size(), prev_output_factors, cur_output_factors.size(), cur_output_factors))
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens,cur_output_factors, encoder_out)
        decoder_b_out = self.decoder_b(prev_output_factors, encoder_out)
        return decoder_out, decoder_b_out


@register_model_architecture('lstm_two_decoders_async', 'lstm_two_decoders_async')
def lstm_two_decoders(args):
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 1024)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 1024)
    args.encoder_bidirectional= getattr(args, 'encoder_bidirectional', True)
