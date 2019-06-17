import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.modules import AdaptiveSoftmax
from fairseq.models import (
    FairseqEncoder, FairseqIncrementalDecoder, BaseFairseqModel, register_model,
    register_model_architecture,
)
from fairseq.models.lstm import LSTMModel,LSTMEncoder,LSTMDecoder,base_architecture


@register_model('bahdanau_rnn')
class BahdanauRNNModel(LSTMModel):

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
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
        )
        return cls(encoder, decoder)


class GRUEncoder(FairseqEncoder):
    """GRU encoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        left_pad=True, pretrained_embed=None, padding_value=0.,
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        #Here the LSTM overwrote the initialization of weights
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
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
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply GRU
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        packed_outs, final_hiddens = self.rnn(packed_x, h0)

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            #shape: (num_layers, bsz, 2*hidden_size)


        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
        #shape: (seq_len,bsz)

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

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x source_embed_dim

        #Repeat TL hidden state srclen times
        # x: srclen x bsz x input_embed_dim
        x=self.dropout_input(input).expand(source_hids.size(0),-1,-1)

        #Apply same dropout to all timesteps of source_hids, like in Nematus
        mask = Bernoulli(torch.full_like(source_hids[0], 1 - self.dropout_p)).sample()/(1 - self.dropout_p)
        source_hids=source_hids*mask # In theory, with broadcasting we multiply all timesteps by the same mask

        #Concatenate with source_hids
        x=torch.cat((x,source_hids),-1)

        #Apply linear layer +  tanh
        x=self.activ_input_proj(self.input_proj(x))
        #x: srclen x bsz x alignment_dim

        #Reduce to one score
        x= torch.squeeze(self.score_proj(x),-1)
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
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        return x, attn_scores

class GRUDecoder(FairseqIncrementalDecoder):
    """GRU decoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size:
            self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        #linear + tanh for initial state
        #TODO: we are assuming encoder is always bidirectional
        #TODO: Linear ignores dropout!!
        self.linear_initial_state=Linear(encoder_output_units*2,hidden_size,dropout=dropout_in)
        self.activ_initial_state=nn.Tanh()

        #TODO: should we apply droput here?
        self.layers = nn.ModuleList([
            # LSTM used custom initialization here
            nn.GRUCell(
                input_size=hidden_size + embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        if attention:
            # TODO make bias configurable
            self.attention = ConcatAttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=True, dropout=dropout_out)#bias = True like Bahdanau
        else:
            self.attention = None

        #Deep output
        self.logit_lstm=Linear(hidden_size, out_embed_dim, dropout=dropout_out)
        self.logit_prev=Linear(out_embed_dim, out_embed_dim, dropout=dropout_out)
        self.logit_ctx=Linear(2*encoder_output_units, out_embed_dim, dropout=dropout_out)
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

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, encoder_hiddens = encoder_out[:2]
        srclen = encoder_outs.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
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

            avg_states=torch.div( torch.sum(encoder_outs,0) , torch.sum(encoder_padding_mask,0).unsqueeze(1)    )
            #shape: (bsz,num_directions*hidden_size)
            hidden=self.activ_initial_state(self.linear_initial_state(avg_states))
            #shape: (bsz,decoder_hidden_size)

            #TODO: All layers have the same initial state: problematic!!!!
            prev_hiddens=[hidden for i in range(num_layers)]

        attn_scores = x.new_zeros(srclen, seqlen, bsz)
        outs = []
        context_vectors=[]
        for j in range(seqlen):
            # apply attention using the last layer's hidden state
            if self.attention is not None:
                context_vector, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
            else:
                context_vector = hidden
            context_vectors.append(context_vector)

            #TODO: apply this dropout?
            #out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input to GRU: concatenate context vector and input embeddings
            input = torch.cat((x[j, :, :], context_vector), dim=1)

            for i, rnn in enumerate(self.layers):
                #TODO: think about dropout
                # recurrent cell:
                hidden = rnn(input, prev_hiddens[i])

                # hidden state becomes the input to the next layer
                #input = F.dropout(hidden, p=self.dropout_out, training=self.training)
                #This is not needed anymore: input feeding is disabled in Bahdanau

                # save state for next time step
                prev_hiddens[i] = hidden

            out = F.dropout(hidden, p=self.dropout_out, training=self.training)

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
        x=self.activ_deep_output(self.logit_ctx() + self.logit_prev(logit_prev_input) + self.logit_lstm(x) )

        # project back to size of vocabulary
        if self.adaptive_softmax is None:
            if hasattr(self, 'additional_fc'):
                x = self.additional_fc(x)
                x = F.dropout(x, p=self.dropout_out, training=self.training)
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
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

def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    d=nn.Dropout(dropout)
    return nn.Sequential(m,d)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


@register_model_architecture('bahdanau_rnn', 'bahdanau_rnn')
def bahdanau_rnn(args):
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 1024)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 1024)
    args.encoder_bidirectional= getattr(args, 'encoder_bidirectional', True)
