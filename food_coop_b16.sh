#!/bin/bash

# custom config
DATA="/localtmp/ktm8eh/datasets/" # "/localtmp/ktm8eh/datasets/imagenet1k"

TRAINER=CoOp

DATASET=food101
CFG=vit_b16  # config file
CTP=end  # class token position (end or middle)
NCTX=8  # number of context tokens
SHOTS=4  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
alpha=0.0
beta=1.0
N_CTX_LABEL=4
N_CTX_DESCRIPTOR=4


# /coop
for SEED in 1 2 3
do
    DIR=output/${DATASET}/coop/coop/${CFG}_${SHOTS}shots/nctx${N_CTX_LABEL}${N_CTX_DESCRIPTOR}_csc${CSC}_ctp${CTP}/bce${alpha}ce${beta}/seed${SEED}_weighted
    rm -r "$DIR"
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
      ! CUDA_VISIBLE_DEVICES=8 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.COOP.alpha ${alpha} \
        TRAINER.COOP.beta ${beta} \
        TRAINER.COOP.N_CTX_LABEL ${N_CTX_LABEL} \
        TRAINER.COOP.N_CTX_DESCRIPTOR ${N_CTX_DESCRIPTOR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATALOADER.TRAIN_X.BATCH_SIZE 128 \
        OPTIM.MAX_EPOCH 200
    fi
done

#!/bin/bash

# custom config
DATA="/localtmp/ktm8eh/datasets/" # "/localtmp/ktm8eh/datasets/imagenet1k"

TRAINER=CoOp

DATASET=food101
CFG=vit_b16  # config file
CTP=end  # class token position (end or middle)
NCTX=8  # number of context tokens
SHOTS=8  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
alpha=0.0
beta=1.0
N_CTX_LABEL=4
N_CTX_DESCRIPTOR=4


# /coop
for SEED in 1 2 3
do
    DIR=output/${DATASET}/coop/coop/${CFG}_${SHOTS}shots/nctx${N_CTX_LABEL}${N_CTX_DESCRIPTOR}_csc${CSC}_ctp${CTP}/bce${alpha}ce${beta}/seed${SEED}_weighted
    rm -r "$DIR"
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
      ! CUDA_VISIBLE_DEVICES=8 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.COOP.alpha ${alpha} \
        TRAINER.COOP.beta ${beta} \
        TRAINER.COOP.N_CTX_LABEL ${N_CTX_LABEL} \
        TRAINER.COOP.N_CTX_DESCRIPTOR ${N_CTX_DESCRIPTOR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATALOADER.TRAIN_X.BATCH_SIZE 128 \
        OPTIM.MAX_EPOCH 200
    fi
done


#!/bin/bash

# custom config
DATA="/localtmp/ktm8eh/datasets/" # "/localtmp/ktm8eh/datasets/imagenet1k"

TRAINER=CoOp

DATASET=food101
CFG=vit_b16  # config file
CTP=end  # class token position (end or middle)
NCTX=8  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
alpha=0.0
beta=1.0
N_CTX_LABEL=4
N_CTX_DESCRIPTOR=4


# /coop
for SEED in 1 2 3
do
    DIR=output/${DATASET}/coop/coop/${CFG}_${SHOTS}shots/nctx${N_CTX_LABEL}${N_CTX_DESCRIPTOR}_csc${CSC}_ctp${CTP}/bce${alpha}ce${beta}/seed${SEED}_weighted
    rm -r "$DIR"
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
      ! CUDA_VISIBLE_DEVICES=8 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.COOP.alpha ${alpha} \
        TRAINER.COOP.beta ${beta} \
        TRAINER.COOP.N_CTX_LABEL ${N_CTX_LABEL} \
        TRAINER.COOP.N_CTX_DESCRIPTOR ${N_CTX_DESCRIPTOR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATALOADER.TRAIN_X.BATCH_SIZE 128 \
        OPTIM.MAX_EPOCH 200
    fi
done



