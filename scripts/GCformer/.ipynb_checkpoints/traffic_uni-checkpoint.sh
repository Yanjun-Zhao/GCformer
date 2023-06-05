if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=PatchTST
root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom
random_seed=2023


seq_len=336
context_len=336
pred_len=96   
python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --context_len $context_len\
      --enc_in 1  \
      --enc_raw 1 \
      --h_channel 862\
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.5\
      --fc_dropout 0.5\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --lradj 'TST'\
      --pct_start 0.2\  
      --train_epochs 100\
      --patience 3\
      --revin 0\
      --itr 1\
      --local_bias 2\
      --global_bias 2\
      --batch_size 2 \
      --learning_rate 0.0005 --decomposition 0  --weight_decay 0.0001 --gpu 1 --decoder_type 'token' --norm_type 'revin'
      #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log #2>&1 &

