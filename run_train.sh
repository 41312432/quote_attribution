ROOT_DIR="/home/wansik/"
BERT_PRETRAINED_DIR="bert-base-uncased"
CHECKPOINT_DIR="/home/wansik/QA/CandidateScoreNet_with_revision/checkpoint"
DATA_PREFIX="/home/wansik/QA/CandidateScoreNet_with_revision/data"

source ${ROOT_DIR}.bashrc

CUDA_VISIBLE_DEVICES=0 python3 train.py \
--model_name CSN \
--pooling_type max_pooling \
--dropout 0.5 \
--optimizer adam \
--margin 1.0 \
--lr 2e-5 \
--num_epochs 50 \
--batch_size 16 \
--patience 10 \
--bert_pretrained_dir ${BERT_PRETRAINED_DIR} \
--train_file \
${DATA_PREFIX}/train/pnp_train.txt \
--val_file \
${DATA_PREFIX}/val/pnp_val.txt \
--test_file \
${DATA_PREFIX}/test/pnp_test.txt \
--name_list \
${DATA_PREFIX}/pnp_name_list.txt \
--len_limit 510 \
--checkpoint_dir ${CHECKPOINT_DIR}