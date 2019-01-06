python -m elmoformanylangs.biLM train \
    --train_path twitter_750mb_en_data.raw \
    --config_path config.json \
    --model model \
    --optimizer adam \
    --lr 0.01 \
    --lr_decay 0.8 \
    --max_sent_len 20 \
    --max_vocab_size 150000 \
    --min_count 3 \
    --max_epoch 10 \
    --batch_size 32 \
    --gpu 0