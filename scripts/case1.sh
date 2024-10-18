# Note: the order of cases is inconsistent with the presentation in the paper.
# train with 19 ppl, test with another 1 ppl
declare -a array=(1 3 5 7 9 11)
for idx in "${array[@]}"
do 
    python main.py \
        --case 1 \
        --epochs 100 \
        --learning_rate 0.1 \
        --round 100 \
        --ppl_idx ${idx}\
        --train_mode \
        --use_mad \
        --data_saved # using the existing built data
done