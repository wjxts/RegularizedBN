#!/usr/bin/env bash

encoder_att_norm=$1
encoder_ffn_norm=$1
encoder_fc1=nnlinear
encoder_fc2=nnlinear
epoch=60
seed=1
folder=transformer_post_max_epoch_seed$seed\_$encoder_att_norm\_$encoder_ffn_norm\_$encoder_fc1
num=6
fairseq-train \
data-bin/iwslt14.tokenized.de-en \
--seed $seed \
--arch transformer_iwslt_de_en --share-all-embeddings \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
--dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
--lr-scheduler inverse_sqrt  --warmup-updates 8000 \
--lr 0.0005  --keep-last-epochs 7 \
--label-smoothing 0.1 --weight-decay 0.0001 \
--max-tokens 4096 --save-dir checkpoints/$folder \
--update-freq 1  --log-interval 50 \
--max-epoch $epoch \
--restore-file checkpoints/$folder/checkpoint_best.pt \
--criterion label_smoothed_cross_entropy  \
--encoder-att-norm $encoder_att_norm --encoder-ffn-norm $encoder_ffn_norm \
--encoder-fc1 $encoder_fc1 --encoder-fc2 $encoder_fc2 

python scripts/average_checkpoints.py \
--inputs checkpoints/$folder/ \
--num-epoch-checkpoints $num  --output checkpoints/$folder/averaged_model.pt  

fairseq-generate \
data-bin/iwslt14.tokenized.de-en/ --path ./checkpoints/$folder/averaged_model.pt \
--remove-bpe --beam 5 --batch-size 64 --lenpen 1 --quiet \
--max-len-a 1 --max-len-b 50|tee generate.out