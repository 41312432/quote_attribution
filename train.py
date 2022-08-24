
import time
import json
import os
import logging
from torch.utils.tensorboard import SummaryWriter


import numpy as np
import torch
import torch.nn as nn

from preprocessing.data_preprocessing import build_data_loader

from utils.arguments import get_train_args

from transformers import BertTokenizer
from model.model import CSN

from tqdm import tqdm


#----------import section----------


#training log
LOG_FORMAT = '%(asctime)s %(name)s %(levelname)s %(pathname)s %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%m:%s %a'

def train():
    """
    Training script.

    return
        best_val_acc
        best_test_acc
    """

    args = get_train_args()
    
    timestamp = time.strftime("%y%m%d%H%M", time.localtime()) #for checkpoint

    print("######Options######")
    print(json.dumps(vars(args), indent=4))

    #checkpoint
    checkpoint_dir = os.path.join(args.checkpoint_dir, os.path.join(args.model_name, timestamp))

    #log
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'tensorboard'))
    logging.basicConfig(level=logging.INFO,
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT,
                        filename=os.path.join(checkpoint_dir, 'training_log.log'))

    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #data
    train_file = args.train_file
    val_file = args.val_file
    test_file = args.test_file
    name_list = args.name_list

    """
    alias_to_id -> Dict : {alias:char_id(int)}
    """
    with open(name_list, 'r', encoding='utf-8') as fin:
        name_lines = fin.readlines()

    alias_to_id = {}
    for i, line in enumerate(name_lines):
        for alias in line.strip().split(';'):
            alias_to_id[alias.lower()] = i

    print("alias2id : \n", alias_to_id)
    
    #data_loaders
    train_data = build_data_loader(train_file, alias_to_id, args)
    val_data = build_data_loader(val_file, alias_to_id, args)
    test_data = build_data_loader(test_file, alias_to_id, args)

    print('\n##############VAL EXAMPLE#################\n')
    val_test_iter = iter(val_data)
    val_test_iter.next()
    tokenized_sentences, candidate_specific_segements, sentence_lens, mention_positions, quote_indicies, one_hot_label, true_index= val_test_iter.next()
    print('Tokenized Sentences :')
    print(tokenized_sentences)
    print('Candidate-specific segments:')
    for css in candidate_specific_segements:
        print('\n', css)
    print('Sentence Length:')
    print(sentence_lens)
    print('Mention Positions:')
    print(mention_positions)
    print('Quote Indices:')
    print(quote_indicies)
    print('one-hot-label:')
    print(one_hot_label)
    print('true index:')
    print(true_index)
    print('\n##############VAL EXAMPLE#################\n')

    # tokenizer = BertTokenizer.from_pretrained(args.bert_pretrained_dir)
    # model = CSN(args)
    # model = model.to(device)

    # # initialize optimizer
    # if args.optimizer == 'sgd':
    #     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # elif args.optimizer == 'adam':
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # else:
    #     raise ValueError("Unknown optimizer type...")

    # # loss
    # loss_function = nn.MarginRankingLoss()

    # print('\n##############TRAIN BEGIN#################\n')

    # #Logging
    # best_val_acc = 0
    # best_val_loss = 0
    # new_best = False

    # #TRAIN
    # for epoch in tqdm(args.num_epochs):
    #     acc_numerator = 0
    #     acc_denominator = 0
    #     train_loss = 0

    #     model.train()
    #     optimizer.zero_grad()

    #     print(f'Epoch: {epoch+1}')
    #     for i, (_, )

if __name__ == '__main__':
    # run several times and calculate average accuracy and standard deviation
    train()
    # val = []
    # test = []
    # for i in range(3):    
    #     val_acc, test_acc = train()
    #     val.append(val_acc)
    #     test.append(test_acc)

    # val = np.array(val)
    # test = np.array(test)

    # val_mean = np.mean(val)
    # val_std = np.std(val)
    # test_mean = np.mean(test)
    # test_std = np.std(test)

    # print(str(val_mean) + '(±' + str(val_std) + ')')
    # print(str(test_mean) + '(±' + str(test_std) + ')')