import copy

from tqdm import tqdm
from transformers import AutoTokenizer

from torch.utils.data import Dataset, DataLoader

def locate_nearest_mention(tokenized_sentences, mention_positions, window_size):
    """
    locate NearestMentionLocation

    Parameter:
        tokenized_sentences
        mention_positions
        window_size
    Return:
        position of mention that is the nearest to Quote
    """
    def word_distance(position):
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
        #의문점 : 근데 이러면 각 quote마다 주위 20개를 보고 그 20개의 모든 토큰들에 대해서 alias에 찍어봐야되는데 너무 timecomplexity 걸리는거 아닌지

    return tokenized_sentences, character_mention_positions

def create_candidate_specific_segments(tokenized_sentences, candidate_mention_positions, window_size, max_len):
    """
    Create Candidate-Specific-Segments for each candidate in instance

    Parameter:
        tokenized_sentences
        candidate_mention_positions
        window_size - 10
        max_len
    
    Return
        
    """

    def max_len_cut(tokenized_sentences, mention_positions):
        """
        Parameter:
            tokenized_sentences
            mention_positions
        
        Return
            
        
        """
        #각 문장 길이
        sentence_lens = [sum(len(word) for word in sentence) for sentence in tokenized_sentences]
        #instance 총길이
        instance_len = sum(sentence_lens)

        blank_index = [len(sentence)-1 for sentence in tokenized_sentences]

        #TODO : UNDERSTANDIT
        while instance_len > max_len:
            longest_sentence_index = max(list(enumerate(sentence_lens)), key=lambda x: x[1])[0]

            if longest_sentence_index == mention_positions[0] and blank_index[longest_sentence_index] == mention_positions[1]:
                blank_index[longest_sentence_index] -= 1
            
            if longest_sentence_index == mention_positions[0] and blank_index[longest_sentence_index] < mention_positions[1]:
                mention_positions[1] -= 1
            
            #TODO : RENAMEIT
            reduced_char_len = len(tokenized_sentences[longest_sentence_index][blank_index[longest_sentence_index]])
            sentence_lens[longest_sentence_index] -= reduced_char_len
            instance_len = sum(sentence_lens)

            del tokenized_sentences[longest_sentence_index][blank_index[longest_sentence_index]]

            blank_index[longest_sentence_index] -= 1

        return tokenized_sentences, mention_positions
    
    #TODO : Naming
    r_candidate_specific_segments = []
    r_sentence_lens = []
    r_mention_positions = []
    r_quote_indices = []

    for candidate_index in candidate_mention_positions.keys():
        nearest_position = locate_nearest_mention(tokenized_sentences, candidate_mention_positions[candidate_index], window_size)
        if nearest_position[0] <= window_size:
            candidate_specific_segment = copy.deepcopy(tokenized_sentences[nearest_position[0] : window_size+1])
            mention_position = [0, nearest_position[1]]
            quote_index = window_size - nearest_position[0]
        else:
            candidate_specific_segment = copy.deepcopy(tokenized_sentences[window_size : nearest_position[0]+1])
            mention_position = [nearest_position[0]-window_size, nearest_position[1]]
            quote_index = 0
        
        cut_candidate_specific_segment, mention_position = max_len_cut(candidate_specific_segment, mention_position)

        sentence_lens = [sum(len(word) for word in sentence) for sentence in cut_candidate_specific_segment]

        mention_position_left = sum(sentence_lens[ : mention_position[0]]) + sum(len(x) for x in cut_candidate_specific_segment[mention_position[0]][ : mention_position[1]])
        mention_position_right = mention_position_left + len(cut_candidate_specific_segment[mention_position[0]][mention_position[1]])
        mention_position = (mention_position[0], mention_position_left, mention_position_right)
        
        r_candidate_specific_segments.append(''.join([''.join(sentence) for sentence in cut_candidate_specific_segment]))
        r_sentence_lens.append(sentence_lens)
        r_mention_positions.append(mention_position)
        r_quote_indices.append(quote_index)
    
    return r_candidate_specific_segments, r_sentence_lens, r_mention_positions, r_quote_indices
    

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
        data_file:
        alias_to_id:
        args:
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

            if not candidate_mention_positions:
                print("NO MENTION")
            
            #mention이 1개밖에 없다면 (근데 이게 mention인지, 그냥 대화 중간에 있는건지, 뭐 그냥 언급인지 어떻게 알지? 에초에 NML이 왜 되는거지)
            if skip_only_one and len(candidate_mention_positions) == 1:
                continue
        
            candidate_specific_segements, sentence_lens, mention_positions, quote_indicies = \
            create_candidate_specific_segments(tokenized_sentences, candidate_mention_positions, args.window_size, args.length_limit)

            one_hot_label = [0 if character_index != alias_to_id[speaker_name.lower()] else 1 for character_index in candidate_mention_positions.keys()]
            true_index = one_hot_label.index(1) if 1 in one_hot_label else 0

            data_list.append((tokenized_sentences, candidate_specific_segements, sentence_lens, mention_positions, quote_indicies, one_hot_label, true_index))

    return DataLoader(InstanceDataSet(data_list), batch_size=1, collate_fn=lambda x: x[0])
import copy

from tqdm import tqdm
from transformers import AutoTokenizer

from torch.utils.data import Dataset, DataLoader

def locate_nearest_mention(tokenized_sentences, mention_positions, window_size):
    """
    locate NearestMentionLocation

    Parameter:
        tokenized_sentences
        mention_positions
        window_size
    Return:
        position of mention that is the nearest to Quote
    """
    def word_distance(position):
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
        #의문점 : 근데 이러면 각 quote마다 주위 20개를 보고 그 20개의 모든 토큰들에 대해서 alias에 찍어봐야되는데 너무 timecomplexity 걸리는거 아닌지

    return tokenized_sentences, character_mention_positions

def create_candidate_specific_segments(tokenized_sentences, candidate_mention_positions, window_size, max_len):
    """
    Create Candidate-Specific-Segments for each candidate in instance

    Parameter:
        tokenized_sentences
        candidate_mention_positions
        window_size - 10
        max_len
    
    Return
        
    """

    def max_len_cut(tokenized_sentences, mention_positions):
        """
        Parameter:
            tokenized_sentences
            mention_positions
        
        Return
            
        
        """
        #각 문장 길이
        sentence_lens = [sum(len(word) for word in sentence) for sentence in tokenized_sentences]
        #instance 총길이
        instance_len = sum(sentence_lens)

        blank_index = [len(sentence)-1 for sentence in tokenized_sentences]

        #TODO : UNDERSTANDIT
        while instance_len > max_len:
            longest_sentence_index = max(list(enumerate(sentence_lens)), key=lambda x: x[1])[0]

            if longest_sentence_index == mention_positions[0] and blank_index[longest_sentence_index] == mention_positions[1]:
                blank_index[longest_sentence_index] -= 1
            
            if longest_sentence_index == mention_positions[0] and blank_index[longest_sentence_index] < mention_positions[1]:
                mention_positions[1] -= 1
            
            #TODO : RENAMEIT
            reduced_char_len = len(tokenized_sentences[longest_sentence_index][blank_index[longest_sentence_index]])
            sentence_lens[longest_sentence_index] -= reduced_char_len
            instance_len = sum(sentence_lens)

            del tokenized_sentences[longest_sentence_index][blank_index[longest_sentence_index]]

            blank_index[longest_sentence_index] -= 1

        return tokenized_sentences, mention_positions
    
    #TODO : Naming
    r_candidate_specific_segments = []
    r_sentence_lens = []
    r_mention_positions = []
    r_quote_indices = []

    for candidate_index in candidate_mention_positions.keys():
        nearest_position = locate_nearest_mention(tokenized_sentences, candidate_mention_positions[candidate_index], window_size)
        if nearest_position[0] <= window_size:
            candidate_specific_segment = copy.deepcopy(tokenized_sentences[nearest_position[0] : window_size+1])
            mention_position = [0, nearest_position[1]]
            quote_index = window_size - nearest_position[0]
        else:
            candidate_specific_segment = copy.deepcopy(tokenized_sentences[window_size : nearest_position[0]+1])
            mention_position = [nearest_position[0]-window_size, nearest_position[1]]
            quote_index = 0
        
        cut_candidate_specific_segment, mention_position = max_len_cut(candidate_specific_segment, mention_position)

        sentence_lens = [sum(len(word) for word in sentence) for sentence in cut_candidate_specific_segment]

        mention_position_left = sum(sentence_lens[ : mention_position[0]]) + sum(len(x) for x in cut_candidate_specific_segment[mention_position[0]][ : mention_position[1]])
        mention_position_right = mention_position_left + len(cut_candidate_specific_segment[mention_position[0]][mention_position[1]])
        mention_position = (mention_position[0], mention_position_left, mention_position_right)
        
        r_candidate_specific_segments.append(''.join([''.join(sentence) for sentence in cut_candidate_specific_segment]))
        r_sentence_lens.append(sentence_lens)
        r_mention_positions.append(mention_position)
        r_quote_indices.append(quote_index)
    
    return r_candidate_specific_segments, r_sentence_lens, r_mention_positions, r_quote_indices
    

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
        data_file:
        alias_to_id:
        args:
        skip_only_once : 

    Ouput:

    """
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    for alias in alias_to_id:
        tokenizer.add_tokens(alias)

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
            raw_sentences_in_list.append(line.strip())

        # speaker
        if offset == 22:
            speaker_name = line.strip().split()[-1]
            
            tokenized_sentences, candidate_mention_positions = tokenize_and_locate_mention(raw_sentences_in_list, alias_to_id, tokenizer)

            if not candidate_mention_positions:
                print("NO MENTION")
            
            #mention이 1개밖에 없다면 (근데 이게 mention인지, 그냥 대화 중간에 있는건지, 뭐 그냥 언급인지 어떻게 알지? 에초에 NML이 왜 되는거지)
            if skip_only_one and len(candidate_mention_positions) == 1:
                continue
        
            candidate_specific_segements, sentence_lens, mention_positions, quote_indicies = \
            create_candidate_specific_segments(tokenized_sentences, candidate_mention_positions, args.window_size, args.len_limit)

            one_hot_label = [0 if character_index != alias_to_id[speaker_name.lower()] else 1 for character_index in candidate_mention_positions.keys()]
            true_index = one_hot_label.index(1) if 1 in one_hot_label else 0

            data_list.append((tokenized_sentences, candidate_specific_segements, sentence_lens, mention_positions, quote_indicies, one_hot_label, true_index))

    return DataLoader(InstanceDataSet(data_list), batch_size=1, collate_fn=lambda x: x[0])
