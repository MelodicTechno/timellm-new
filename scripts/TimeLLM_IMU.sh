model_name=IMULLM
train_epochs=2
learning_rate=0.01
llama_layers=32

num_process=1
batch_size=2
d_model=32
d_ff=128

comment='TimeLLM-imu01'

accelerate launch --mixed_precision bf16 --num_processes $num_process run_imu.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/imu/ \
  --label_dict ./dataset/label.json \
  --loader modal \
  --freq h \
  --checkpoints ./checkpoints \
  --data_path all_acc_gyro.csv \
  --model_id IMU \
  --model $model_name \
  --data imu01 \
  --features M \
  --seq_len 96 \
  --seasonal_patterns 'Monthly' \
  --label_len 48 \
  --pred_len 96 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 8 \
  --e_layers 2 \
  --d_layers 1 \
  --moving_avg 25 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --seed 2024