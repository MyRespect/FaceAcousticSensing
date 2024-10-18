# Note: the order of cases is inconsistent with the presentation in the paper.
# train without mask, test with mask
python main.py \
    --case 4 \
    --epochs 100 \
    --learning_rate 0.1 \
    --round 150 \
    --train_mode \
    --use_mad \
    --data_saved # using the existing built data