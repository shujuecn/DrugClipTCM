#!/bin/zsh

export PYTHONPATH=$PYTHONPATH:$(pwd)/Uni-Core

BATCH_SIZE=256
NUM_WORKERS=8

DATA_PATH="./data_dict"
WEIGHT_PATH="/mnt/e/Workspace/260217-DrugClip/files/checkpoint_best.pt"
MOL_PATH="/mnt/e/Workspace/260217-DrugClip/files/retrieval/mols.lmdb" # path to the molecule file
POCKET_PATH="/mnt/e/Workspace/260217-DrugClip/files/retrieval/pocket.lmdb" # path to the pocket file

EMB_DIR="./results_new"  # replace to your results path
mkdir -p $EMB_DIR

echo "开始运行 DrugCLIP 演示筛选..."

# conda activate drugclips
CUDA_VISIBLE_DEVICES="0" python ./unimol/retrieval.py \
       --user-dir ./unimol \
       $DATA_PATH \
       --valid-subset test \
       --results-path $EMB_DIR \
       --num-workers $NUM_WORKERS \
       # --ddp-backend=c10d \
       --batch-size $BATCH_SIZE \
       --task drugclip \
       # --loss in_batch_softmax \
       --arch drugclip  \
       # --max-pocket-atoms 256 \
       --fp16 \
       # --fp16-init-scale 4 \
       # --fp16-scale-window 256 \
       # --seed 1 \
       --path $WEIGHT_PATH \
       # --log-interval 100 \
       # --log-format simple \
       --mol-path $MOL_PATH \
       --pocket-path $POCKET_PATH \
       --emb-dir $EMB_DIR \

echo "筛选完成！结果已保存在 $EMB_DIR/ranked_compounds.txt"
