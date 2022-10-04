from math import fabs
import torch.nn as nn
import torch
from transformers import AutoModel, TransfoXLModel

def get_nonlinear(nonlinear):
    """
    Activation function.
    """
    nonlinear_dict = {'relu':nn.ReLU(inplace=True), 'tanh':nn.Tanh(), 'sigmoid':nn.Sigmoid(), 'softmax':nn.Softmax(dim=-1)}
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
        self.linear_1 = nn.Linear(classifier_input_size, args.classifier_intermediate_dim)
        self.linear_2 = nn.Linear(args.classifier_intermediate_dim, 1)
        self.nonlinear = get_nonlinear(args.nonlinear_type)

    def forward(self, x):
        x = self.linear_1(x).clone().detach()
        x = self.nonlinear(x)
        x = self.linear_2(x)
        x = self.nonlinear(x)

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
        self.mlp_scorer = MLP_Scorer(args, self.model.config.hidden_size * 3)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, features, quote_indices, context_indices, mention_indices, true_index, device):
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
        mention_hidden = [] #mention

        for i, (cdd_quote_idx, cdd_context_idx, cdd_mention_idx) in enumerate(zip(quote_indices, context_indices, mention_indices)):

            model_output = self.model(torch.tensor([features[i].input_ids]).to(device))

            CSS_hidden = model_output['last_hidden_state'][0][1:]

            quote_hidden.append(CSS_hidden[cdd_quote_idx[0]:cdd_quote_idx[1]])
            context_hidden.append(CSS_hidden[cdd_context_idx[0]:cdd_context_idx[1]])
            mention_hidden.append(CSS_hidden[cdd_mention_idx].unsqueeze(0))
        # pooling
        quote_rep = self.pooling(quote_hidden)
        context_rep = self.pooling(context_hidden)
        mention_rep = self.pooling(mention_hidden)

        # # concatenate
        feature_vector = torch.cat([quote_rep, context_rep, mention_rep], dim=-1)

        # # dropout
        feature_vector = self.dropout(feature_vector)

        # # scoring
        scores = self.mlp_scorer(feature_vector).view(-1)
        scores_false = [scores[i] for i in range(scores.size(0)) if i != true_index]
        scores_true = [scores[true_index] for i in range(scores.size(0) - 1)]

        del feature_vector

        return scores, scores_false, scores_true