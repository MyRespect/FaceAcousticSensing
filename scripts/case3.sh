# Note: the order of cases is inconsistent with the presentation in the paper.
# train with 10 ppl, test with another 10 ppl
python main.py \
    --case 3 \
    --epochs 100 \
    --learning_rate 0.1 \
    --round 150 \
    --train_mode \
    --use_mad \
    --data_saved # using the existing built data