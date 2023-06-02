if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=336
context_len=336
pred_len=96
python -u run_longExp.py \
      --random_seed 2021 \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path electricity.csv \
      --model_id Electricity'_'$seq_len'_'$pred_len \
      --model PatchTST \
      --data custom \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --context_len $context_len\
      --enc_in 1 \
      --enc_raw 1 \
      --e_layers 3 \
      --global_layers 3\
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 3\
      --revin 0\
      --itr 1\
      --local_bias 1\
      --global_bias 0\
      --batch_size 16 \
      --learning_rate 0.001 --decomposition 0  --weight_decay 0.001 --gpu 1 --decoder_type 'token' --norm_type 'revin' 
      #>logs/LongForecasting/PatchTST'_'Electricity'_'$seq_len'_'$pred_len.log 
