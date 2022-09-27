
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
from model.xl import *

from transformers import TransfoXLModel, TransfoXLTokenizer
from model.model import CSN

from tqdm import tqdm

#----------import section----------

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train():
    """
    Training script.

    return
        best_val_acc
        best_test_acc
    """

    args = get_train_args()

    timestamp = time.strftime("%y%m%d%H%M", time.localtime()) #for checkpoint

    checkpoint_dir = os.path.join(args.checkpoint_dir, 
                                  os.path.join(args.model_name, timestamp))

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

    #tokenizer
    tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103', lower_case=True)

    for alias in alias_to_id:
        tokenizer.add_tokens(alias)

    #model
    model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
    model.resize_token_embeddings(tokenizer.vocab_size+len(alias_to_id))

    #data_loaders
    train_data = build_data_loader(train_file, alias_to_id, args, tokenizer)
    val_data = build_data_loader(val_file, alias_to_id, args, tokenizer)
    test_data = build_data_loader(test_file, alias_to_id, args, tokenizer)

    print('\n##############VAL EXAMPLE#################\n')
    val_test_iter = iter(val_data)
    val_test_iter.next()
    tokenized_sentences, candidate_specific_segements, quote_indicies, context_indices, mention_indices, one_hot_label, true_index = val_test_iter.next()
    print('Tokenized Sentences :')
    print(tokenized_sentences)
    print('\nCandidate-specific segments:')
    for i, css in enumerate(candidate_specific_segements):
        print(f'\n{i} : {css}')
        print(f'q:{quote_indicies[i]}, c:{context_indices[i]}, m:{mention_indices[i]}')

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

    best_val_acc = 0
    best_val_loss = 0
    new_best = False

    patience_counter = 0
    backward_counter = 0

    #TRAIN
    for epoch, _ in enumerate(tqdm(range(args.num_epochs))):
        acc_numerator = 0
        acc_denominator = 0
        train_loss = 0

        model.train()
        optimizer.zero_grad()

        print(f'#########\nEpoch: {epoch+1} begins:\n')
        for i, (_, candidate_specific_segements, quote_indicies, context_indices, mention_indices, _ , true_index) in enumerate(tqdm(train_data)):
            try:
                #candidate_specific_segments : instance의 css들 list
                features = convert_examples_to_features(examples=candidate_specific_segements, tokenizer=tokenizer)
                # print(features[0].tokens[mention_indices[0]+1])
                # print(features[0].tokens[context_indices[0][0]+1:context_indices[0][1]+1])
                # print(features[0].tokens[quote_indicies[0][0]+1:quote_indicies[0][1]+1])
                scores, scores_false, scores_true = model(features, mention_positions, quote_indicies, true_index, device)

                for (false, true) in zip(scores_false, scores_true):
                    #Loss
                    loss = loss_function(false.unsqueeze(0), true.unsqueeze(0), torch.tensor(-1.0).unsqueeze(0).to(device))
                    train_loss += loss.item()

                    #Back Propagation
                    loss = loss / args.batch_size
                    loss.backward(retain_graph=True)
                    backward_counter += 1

                    #Update Weights
                    if backward_counter % args.batch_size == 0:
                        optimizer.step()
                        optimizer.zero_grad()
            
                #print(scores(max(0)))
                acc_numerator += 1 if scores.max(0)[1].item()==true_index else 0
                #acc_numertate += 1
                acc_denominator += 1

            except Exception as e:
                print(e)
                break
        
        ######Train Finish#####
        train_acc = acc_numerator / acc_denominator
        train_loss = train_loss / len(train_data)

        print(f'{epoch+1} : train acc : {train_acc}, train loss : {train_loss} \n')

        wandb.log(
                {
                    "train_loss" : train_loss,
                    "train_acc" : train_acc,
                    "epoch" : epoch+1
                }
        )

        #Adjust lr
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.lr_decay

        #Evaluation
        model.eval()

        def eval(eval_data, subset_name):
            """
            Evaluate performance on a given subset.

            params
                eval_data: the set of instances to be evaluate on.
                subset_name: the name of the subset for logging.

            return
                acc_numerator_sub: the number of correct predictions.
                acc_denominator_sub: the total number of instances.
                sum_loss: the sum of evaluation loss on positive-negative pairs.
            """
            eval_acc_numerator = 0
            eval_acc_denominator = len(eval_data)

            eval_sum_loss = 0

            for i, (_, candidate_specific_segements, quote_indicies, context_indices, mention_indices, _ , true_index) in enumerate(tqdm(eval_data)):
                with torch.no_grad():
                    features = convert_examples_to_features(examples=candidate_specific_segements, tokenizer=tokenizer)
                    print(features[0].tokens[quote_indicies[0]+1])
                    print(features[0].tokens[context_indices[0][0]+1:context_indices[0][1]+1])
                    print(features[0].tokens[mention_indices[0][0]+1:mention_indices[0]])
                    scores, scores_false, scores_true = model(features, mention_positions, quote_indicies, true_index, device)
                    loss_list = [loss_function(x.unsqueeze(0), y.unsqueeze(0), torch.tensor(-1.0).unsqueeze(0).to(device)) for x, y in zip(scores_false, scores_true)]
                
                eval_sum_loss += sum(x.item() for x in loss_list)

                # evaluate accuracy
                correct = 1 if scores.max(0)[1].item() == true_index else 0
                eval_acc_numerator += correct

            eval_acc = eval_acc_numerator / eval_acc_denominator
            eval_avg_loss = eval_sum_loss / eval_acc_denominator

            wandb.log(
                    {
                    f"{subset_name} acc" : eval_acc,
                    f"{subset_name} loss": eval_avg_loss,
                    "epoch" : epoch+1
                    }
            )
            print(f'{epoch+1} : {subset_name} acc : {eval_acc}, {subset_name} loss : {eval_avg_loss} \n')

            return eval_acc, eval_avg_loss

        # development stage
        val_acc, val_avg_loss = eval(val_data, 'val')

        # # save the model with best performance
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_avg_loss
            
            patience_counter = 0
            new_best = True
        else:
            patience_counter += 1
            new_best = False

        # only save the model which outperforms the former best on development set
        if new_best:
            # test stage
            test_acc, test_avg_loss = eval(test_data, 'test')
            try:
                try:
                    os.makedirs(checkpoint_dir)
                except Exception as e:
                    print(e)
                with open(os.path.join(checkpoint_dir, 'info.json'), 'w', encoding='utf-8') as f:
                    json.dump({
                        'args': vars(args),
                        'training_loss': train_loss,
                        'best_val_acc': best_val_acc,
                        'best_overall_dev_loss': best_val_loss,
                        'test_acc': test_acc,
                        'overall_test_loss': test_avg_loss
                        }, f, indent=4)
                torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()}, os.path.join(checkpoint_dir, 'csn.ckpt'))
            except Exception as e:
                print(e)

        # early stopping
        if patience_counter > args.patience:
            print("Early stopping...")
            break

        print('------------------------------------------------------')

    return best_val_acc, test_acc                

if __name__ == '__main__':
    # run several times and calculate average accuracy and standard deviation
    wandb.init(project="QA-test")
    wandb.config.update(get_train_args())

    val_acc, test_acc = train()

    print('final :', val_acc, test_acc)
