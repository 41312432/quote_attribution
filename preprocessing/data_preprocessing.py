import copy

from tqdm import tqdm
from transformers import AutoTokenizer

from torch.utils.data import Dataset, DataLoader

def locate_nearest_mention(tokenized_sentences, mention_positions, window_size):
    """
    locate NearestMentionLocation

    Parameter:
        tokenized_sentences
        mention_positions -> List : mention position [sentence_index, word_index_in_sentence] for specific character
        window_size
    Return:
        position of mention that is the nearest to Quote -> list ()
    """
    def word_distance(position):
        #Because of Bert Tokenizer, the word distance lost meanings,,, but for specific character, it still has maybe
        """
        distance(for word size) between Quote and Mention position

        Parameter:
            position : [sentence_index, word_index_in_sentence] for character contain sentence
        Return:
            distance
        """

        #mention in quote
        if position[0] == window_size:
            return window_size*2
        
        #mention before quote
        elif position[0] < window_size:
            return sum(len(sentence) for sentence in tokenized_sentences[position[0]+1 : window_size]) + len(tokenized_sentences[position[0]][position[1]+1 : ])
        
        #mention after quote
        else:
            return sum(len(sentence) for sentence in tokenized_sentences[window_size+1 : position[0]]) + len(tokenized_sentences[position[0]][ : position[1]])

    return sorted(mention_positions, key=lambda x: word_distance(x))[0]

def tokenize_and_locate_mention(raw_sentences_in_list, alias_to_id, tokenizer):
    """
    Tokenize & Locate Character Mention

    Parameter:
        raw_sentences_in_list -> List : in data file, the original(raw) sentences 1~21
        alias_to_id -> Dict : char alias to id

    Return
        tokenized_sentences -> List : list of tokens of input sentence of raw_sentences 1~21
        character_mention_positions -> Dict : {char_id : [[sentence_index, word_index_in_sentence], [sentence_index, word_index_in_sentence], ...], 
                                               char_id : [[sentence_index, word_index_in_sentence], [sentence_index, word_index_in_sentence], ...],
                                               ...}
    """
    tokenized_sentences = []
    character_mention_positions = {}

    for sentence_index, sentence in enumerate(raw_sentences_in_list):
        #sentence tokenize
        tokenized_sentence = tokenizer.tokenize(sentence)
        tokenized_sentences.append(tokenized_sentence)

        #make mention positions for each characters
        for token_index, token in enumerate(tokenized_sentence):
            if token in alias_to_id:
                if alias_to_id[token] in character_mention_positions:
                    character_mention_positions[alias_to_id[token]].append([sentence_index, token_index])
                else:
                    character_mention_positions[alias_to_id[token]]=[[sentence_index, token_index]]

    return tokenized_sentences, character_mention_positions

def create_candidate_specific_segments(tokenized_sentences, candidate_mention_positions, window_size, max_len):
    """
    Create Candidate-Specific-Segments for each candidate in instance

    Parameter:
        tokenized_sentences -> List : list of tokens of input sentence of raw_sentences 1~21
        candidate_mention_positions -> List : Dict : {char_id : [[sentence_index, word_index_in_sentence], [sentence_index, word_index_in_sentence], ...], 
                                               char_id : [[sentence_index, word_index_in_sentence], [sentence_index, word_index_in_sentence], ...],
                                               ...}
        window_size - 10
        max_len
    
    Return
        
    """

    #TODO : Naming
    r_candidate_specific_segments = []
    r_mention_positions = []
    r_quote_indices = []

    candidate_specific_segments = []
    mention_positions = []
    quote_indices = []

    for character_id in candidate_mention_positions.keys():
        #nearest_position(sentence index, word index) for specific character
        nearest_position = locate_nearest_mention(tokenized_sentences, candidate_mention_positions[character_id], window_size)

        if nearest_position[0] <= window_size:
            #candidate_specific_segment = sentence that contain character neareset position ~ quote
            candidate_specific_segment = copy.deepcopy(tokenized_sentences[nearest_position[0] : window_size+1])
            #if candidate in before quote, 0 is candidate sentence index
            mention_position = [0, nearest_position[1]] 
            
            quote_index = window_size - nearest_position[0]
        else:
            #candidate_specific_segment = quote ~ sentence that contain character neareset position
            candidate_specific_segment = copy.deepcopy(tokenized_sentences[window_size : nearest_position[0]+1])
            
            mention_position = [nearest_position[0]-window_size, nearest_position[1]]
            #if candidate in after quote, 0 is quote
            quote_index = 0

        candidate_specific_segments.append(''.join([''.join(sentence) for sentence in cut_candidate_specific_segment]))
        mention_positions.append(mention_position)
        quote_indices.append(quote_index)
    
    return candidate_specific_segments, mention_positions, quote_indices
    

class InstanceDataSet(Dataset):
    def __init__(self, data_list):
        super(InstanceDataSet, self).__init__()
        self.data = data_list
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_data_loader(data_file, alias_to_id, args, skip_only_one=False):
    """
    Build dataloader

    Parameter:
        data_file -> txt : instance 0~23
        alias_to_id -> List : {alias_1 : char_id_1, alias_2 : char_id_2, alias_3 : char_id_1 ...}
        args: ags
        skip_only_once : 

    Ouput:

    """
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    for alias in alias_to_id:
        print(alias, end='\n')
        tokenizer.add_tokens(alias.lower())

    #load instances from file
    with open(data_file, 'r', encoding='utf-8') as f:
        data_lines = f.readlines()
    #data_liens:List

    #pre-processing
    data_list = []

    for i, line in enumerate(tqdm(data_lines, ncols=100, total=len(data_lines))):
        offset = i%24

        #instant index
        if offset == 0:
            raw_sentences_in_list = []
        
        #line[1]~[10]:context line[11]:quote line[12]~[21]:context
        if offset < 22:
            raw_sentences_in_list.append(line.strip().lower())

        # speaker
        if offset == 22:
            speaker_name = line.strip().split()[-1]
            
            tokenized_sentences, candidate_mention_positions = tokenize_and_locate_mention(raw_sentences_in_list, alias_to_id, tokenizer)

            candidate_specific_segements, mention_positions, quote_indicies = \
            create_candidate_specific_segments(tokenized_sentences, candidate_mention_positions, args.window_size)

            one_hot_label = [0 if character_index != alias_to_id[speaker_name.lower()] else 1 for character_index in candidate_mention_positions.keys()]
            true_index = one_hot_label.index(1) if 1 in one_hot_label else 0

            data_list.append((tokenized_sentences, candidate_specific_segements, mention_positions, quote_indicies, one_hot_label, true_index))

    return DataLoader(InstanceDataSet(data_list), batch_size=1, collate_fn=lambda x: x[0])