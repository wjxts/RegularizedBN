#!/usr/bin/env bash

encoder_att_norm=$1
encoder_ffn_norm=$1
encoder_fc1=nnlinear
encoder_fc2=nnlinear
seed=1
folder=transformer_post_max_epoch_seed$seed\_$encoder_att_norm\_$encoder_ffn_norm\_$encoder_fc1
num=6

fairseq-generate \
data-bin/iwslt14.tokenized.de-en/ --path ./checkpoints/$folder/averaged_model.pt \
--remove-bpe --beam 5 --batch-size 64 --lenpen 1 --quiet \
--max-len-a 1 --max-len-b 50|tee generate.out