from argparse import ArgumentParser

def get_train_args():
    """
    return
        args
    """
    parser = ArgumentParser(description='QA')

    #HEADER
    parser.add_argument('--model_name', type=str, default='CSN')

    #checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='')

    #data
    parser.add_argument('--train_file', type=str, default='')
    parser.add_argument('--val_file', type=str, default='')
    parser.add_argument('--test_file', type=str, default='')
    parser.add_argument('--name_list', type=str, default='')

    parser.add_argument('--window_size', type=int, default=10)

    #model
    parser.add_argument('--pooling_type', type=str, default='max_pooling')
    parser.add_argument('--classifier_intermediate_dim', type=int, default=100)
    parser.add_argument('--nonlinear_type', type=str, default='relu')

    parser.add_argument('--bert_pretrained_dir', type=str, default='')

    #training
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=10)

    args = parser.parse_args()
    return args