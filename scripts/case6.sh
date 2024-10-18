# Note: the order of cases is inconsistent with the presentation in the paper.
# merge all, and split into train-test
python main.py \
    --case 6 \
    --epochs 100 \
    --learning_rate 0.1 \
    --round 150 \
    --train_mode \
    --data_saved # using the existing built data