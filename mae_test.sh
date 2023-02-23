# Dataset can be: isic, eurosat, or bccd

python3 mae_test.py \
--root_data_path "/Datasets" \
--dataset isic \
--test_episodes 30 \
--pretrain_epochs 20 \
--pretrain_iters 50 \
--finetune_epochs 0 \
--finetune_iters 0 \
--shots 5 \
--img_size 224 \
--device "cuda:0"