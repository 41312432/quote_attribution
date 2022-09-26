import torch.nn as nn
import torch
from transformers import AutoModel, TransfoXLModel

def get_nonlinear(nonlinear):
    """
    Activation function.
    """
    nonlinear_dict = {'relu':nn.ReLU(), 'tanh':nn.Tanh(), 'sigmoid':nn.Sigmoid(), 'softmax':nn.Softmax(dim=-1)}
    try:
        return nonlinear_dict[nonlinear]
    except:
        raise ValueError('not a valid nonlinear type!')

class SeqPooling(nn.Module):
    """
    Sequence pooling module.

    Can do max-pooling, mean-pooling and attentive-pooling on a list of sequences of different lengths.
    """
    def __init__(self, pooling_type, hidden_dim):
        super(SeqPooling, self).__init__()
        self.pooling_type = pooling_type
        self.hidden_dim = hidden_dim
        if pooling_type == 'attentive_pooling':
            self.query_vec = nn.parameter.Parameter(torch.randn(hidden_dim))

    def max_pool(self, seq):
        return seq.max(0)[0]

    def mean_pool(self, seq):
        return seq.mean(0)

    def attn_pool(self, seq):
        attn_score = torch.mm(seq, self.query_vec.view(-1, 1)).view(-1)
        attn_w = nn.Softmax(dim=0)(attn_score)
        weighted_sum = torch.mm(attn_w.view(1, -1), seq).view(-1)
        return weighted_sum

    def forward(self, batch_seq):
        pooling_fn = {'max_pooling': self.max_pool,
                      'mean_pooling': self.mean_pool,
                      'attentive_pooling': self.attn_pool}
        pooled_seq = [pooling_fn[self.pooling_type](seq) for seq in batch_seq]
        return torch.stack(pooled_seq, dim=0)

class MLP_Scorer(nn.Module):
    """
    MLP scorer module.

    A perceptron with two layers.
    """
    def __init__(self, args, classifier_input_size):
        super(MLP_Scorer, self).__init__()
        self.scorer = nn.ModuleList()
        self.scorer.append(nn.Linear(classifier_input_size, args.classifier_intermediate_dim))
        self.scorer.append(nn.Linear(args.classifier_intermediate_dim, 1))
        self.nonlinear = get_nonlinear(args.nonlinear_type)

    def forward(self, x):
        for model in self.scorer:
            x = self.nonlinear(model(x))
        ###print('X', x)
        return x

class CSN(nn.Module):
    """
    Candidate Scoring Network.

    It's built on BERT with an MLP and other simple components.
    """
    def __init__(self, args, model):
        super(CSN, self).__init__()
        self.args = args
        self.model = model
        self.pooling = SeqPooling(args.pooling_type, self.model.config.hidden_size)
        self.mlp_scorer = MLP_Scorer(args, self.model.config.hidden_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, features, mention_positions, quote_indicies, true_index, device):
        """
        params
            features: the candidate-specific segments (CSS) converted into the form of BERT input.  
            sent_char_lens: character-level lengths of sentences in CSSs.
                [[character-level length of sentence 1,...] in the CSS of candidate 1,...]
            mention_positions: the positions of the nearest candidate mentions.
                [(sentence-level index of nearest mention in CSS, 
                 character-level index of the leftmost character of nearest mention in CSS, 
                 character-level index of the rightmost character + 1) of candidate 1,...]
            quote_indicies: the sentence-level index of the quotes in CSSs.
                [index of quote in the CSS of candidate 1,...]
            true_index: the index of the true speaker.
            device: gpu/tpu/cpu device.
        """
        
        # encoding
        quote_hidden = [] #quote
        context_hidden = [] #context
        candidate_hidden = [] #mention
        qs_hid = []
        ctx_hid = []
        cdd_hid = []

        css_hidden = []
        for i, (cdd_mention_pos, cdd_quote_idx) in enumerate(zip(mention_positions, quote_indicies)):

            model_output = self.model(torch.tensor([features[i].input_ids], dtype=torch.long).to(device))

            accum_char_len = []#전체 char 길이 [0, 10, 20, 30]요런식으로 (각 세 문장 길이가 10이였다면)

            #그리고 char len 없이 짤라보는거 생각
            # print(cdd_mention_pos, cdd_quote_idx)
            # print(len(model_output['last_hidden_state'][0]))    #instance i의 css마다 [CLS], [SEP] 추가시킨 last hidden state
            #model_output['last_hidden_state'][0] : feature word 길이
            #model_output['last_hidden_state'][0][0] : 한 word는 768차원 tensor

            #css_hidden.append(model_output['last_hidden_state'][0]) #Except [CLS], [SEP]
            css_hidden.append(model_output['last_hidden_state'][0][0])

            # CSS_hid = model_output['last_hidden_state'][0][1:sum(cdd_sent_char_lens) + 1]
            # qs_hid.append(CSS_hid[accum_char_len[cdd_quote_idx]:accum_char_len[cdd_quote_idx + 1]])

            # if len(cdd_sent_char_lens) == 1:
            #     ctx_hid.append(torch.zeros(1, CSS_hid.size(1)).to(device))
            # elif cdd_mention_pos[0] == 0:
            #     ctx_hid.append(CSS_hid[:accum_char_len[-2]])
            # else:
            #     ctx_hid.append(CSS_hid[accum_char_len[1]:])
            
            # cdd_hid.append(CSS_hid[cdd_mention_pos[1]:cdd_mention_pos[2]])

        ###print('hidden', css_hidden, css_hidden[0].size())
        ##css_represent = self.pooling(css_hidden)
        ###
        css_represent = []
        for seq in css_hidden:
            css_represent.append(seq)
        css_represent = torch.stack(css_represent, dim=0)
        print('css_represent', css_represent, css_represent.size())

        # feature_vector = torch.cat([css_represent], dim=-1)
        feature_vector = torch.cat([css_represent], dim=-1)
        ###
        print('feature_vector cat', feature_vector)

        # pooling
        # qs_rep = self.pooling(qs_hid)
        # ctx_rep = self.pooling(ctx_hid)
        # cdd_rep = self.pooling(cdd_hid)

        # # concatenate
        # feature_vector = torch.cat([qs_rep, ctx_rep, cdd_rep], dim=-1)
        #feature_vector = torch.cat([css_represent], dim=-1)

        # # dropout
        # feature_vector = self.dropout(feature_vector)
        feature_vector = self.dropout(feature_vector)
        print('feature_vec', feature_vector, feature_vector.size())
        ###

        # # scoring
        # true index가 안들어온당
        scores = self.mlp_scorer(feature_vector).view(-1)
        print('scores', scores, scores.size())
        scores_false = [scores[i] for i in range(scores.size(0)) if i != true_index]
        print('s false', scores_false, len(scores_false))
        scores_true = [scores[true_index] for i in range(scores.size(0) - 1)]
        print('s truye', scores_true, len(scores_true))

        return scores, scores_false, scores_true