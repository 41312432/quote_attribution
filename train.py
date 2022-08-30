
import time
import json
import os
import logging
import wandb

import numpy as np
import torch
import torch.nn as nn

from preprocessing.data_preprocessing import build_data_loader

from utils.arguments import get_train_args
from model.bert import *

from transformers import AutoTokenizer, AutoModel, TransfoXLModel, TransfoXLTokenizer
from model.model import CSN

from tqdm import tqdm


#----------import section----------

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

    alias_len = len(alias_to_id)

    print("alias2id : \n", alias_to_id)

    #tokenizer
    #tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103', lower_case=True)

    for alias in alias_to_id:
        tokenizer.add_tokens(alias)

    #model
    #bert_model = AutoModel.from_pretrained('bert-base-uncased')
    model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
    model.resize_token_embeddings(tokenizer.vocab_size+alias_len)


    #data_loaders
    train_data = build_data_loader(train_file, alias_to_id, args, tokenizer)
    val_data = build_data_loader(val_file, alias_to_id, args, tokenizer)
    test_data = build_data_loader(test_file, alias_to_id, args, tokenizer)

    print('\n##############VAL EXAMPLE#################\n')
    val_test_iter = iter(val_data)
    val_test_iter.next()
    tokenized_sentences, candidate_specific_segements, mention_positions, quote_indicies, one_hot_label, true_index= val_test_iter.next()
    print('Tokenized Sentences :')
    print(tokenized_sentences)
    print('\nCandidate-specific segments:')
    for i, css in enumerate(candidate_specific_segements):
        print('\ni : ', css)
    print('\nMention Positions:')
    print(mention_positions)
    print('\nQuote Indices:')
    print(quote_indicies)
    print('\none-hot-label:')
    print(one_hot_label)
    print('\ntrue index:')
    print(true_index)
    print('\n##############VAL EXAMPLE#################\n')

    model = CSN(args, model)
    model = model.to(device)

    # initialize optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Unknown optimizer type...")

    # loss
    loss_function = nn.MarginRankingLoss()

    print('\n##############TRAIN BEGIN#################\n')

    #Logging
    best_val_acc = 0
    best_val_loss = 0
    new_best = False

    backward_counter = 0

    #TRAIN
    for epoch, _ in enumerate(tqdm(range(args.num_epochs))):
        acc_numerator = 0
        acc_denominator = 0
        train_loss = 0

        model.train()
        optimizer.zero_grad()

        print(f'Epoch: {epoch+1}')
        for i, (_, candidate_specific_segements, mention_positions, quote_indicies, _, true_index) in enumerate(tqdm(train_data)):
            # for css, mp in zip(candidate_specific_segements, mention_positions):
            #     print(css, mp)
            try:
                features = convert_examples_to_features(examples=candidate_specific_segements, tokenizer=tokenizer)
                for e, feature in enumerate(features):
                    print(i, e, 'tokens : ', feature.tokens, len(feature.tokens))

                #feautre 0도 있는거고 1도 있는거고 csss 길이만큼 있는건가?? 그러네..

                scores, scores_false, scores_true = model(features, mention_positions, quote_indicies, true_index, device)

                # for (false, true) in zip(scores_false, scores_true):
                #     #Loss
                #     loss = loss_function(false.unsqueeze(0), true.unsqueeze(0), torch.tensor(-1.0).unsqueeze(0).to(device))
                #     train_loss += loss.item()

                #     #Back Propagation
                #     loss = loss / args.batch_size
                #     loss.backward(retain_graph=True)
                #     backward_counter += 1

                #     #Update Weights
                #     if backward_counter % args.batch_size == 0:
                #         optimizer.step()
                #         optimizer.zero_grad()
            
                # #print(scores(max(0)))
                # #acc_numerator += 1 if scores.max(0)[1].item()==true_index else 0
                # acc_numertate += 1
                # acc_denominator += 1

            except Exception as e:
                print(e)
                break
        
        ######Train Finish#####
        train_acc = acc_numerator / acc_denominator
        train_loss = train_loss / len(train_data)

        print(f'{epoch+1} : train acc : {train_acc}, train loss : {train_loss} \n')

        wandb.log(
            (
                {
                    "train_loss" : train_loss,
                    "train_acc" : train_acc,
                }
            )
        )

        #Adjust lr
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.lr_decay

        #Evaluation
        model.eval()

if __name__ == '__main__':
    # run several times and calculate average accuracy and standard deviation
    wandb.init(project="QA-test")
    wandb.config.update(get_train_args())
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