if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

torch_compile=0

e_layers=3
n_heads=8
dropout=0.1
max_grad_norm=1
patience=12
random_seed=2024

predictor=Default

data_names=("PEMS/PEMS03.npz" "PEMS/PEMS04.npz" "PEMS/PEMS07.npz" "PEMS/PEMS08.npz" "ETT-small/ETTm1.csv" "ETT-small/ETTm2.csv" "ETT-small/ETTh1.csv" "ETT-small/ETTh2.csv" "weather/weather.csv" "Solar/solar_AL.txt" "electricity/electricity.csv" "traffic/traffic.csv")
data_alias=("PEMS03" "PEMS04" "PEMS07" "PEMS08" "ETTm1" "ETTm2" "ETTh1" "ETTh2" "Weather" "Solar" "ECL" "Traffic")
data_types=("PEMS" "PEMS" "PEMS" "PEMS" "ETTm1" "ETTm2" "ETTh1" "ETTh2" "custom" "Solar" "custom" "custom")
enc_ins=(358 307 883 170 7 7 7 7 21 137 321 862)
batch_sizes=(8 8 8 8 32 32 32 32 32 8 8 8)
grad_accums=(4 4 4 4 1 1 1 1 1 4 4 4)

for i in $(seq 0 11); do
data_name=${data_names[$i]}
data_alias_current=${data_alias[$i]}
data_type=${data_types[$i]}
enc_in=${enc_ins[$i]}
batch_size=${batch_sizes[$i]}
gradient_accumulation=${grad_accums[$i]}

for pred_len in 96 192 336 720
do

for seq_len in 4096 2048 1024 512
do

for model_name in FormerBaseline CATS FITS iTransformer PatchTST DLinear
do

if [ "$model_name" = "CATS" ]; then
    random_drop_training=1
else
    random_drop_training=0
fi

# please adjust the d_model based on the dataset and the model
d_model=128

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $data_name \
  --model_id $random_seed'_'$model_name'_'$predictor'_'$data_alias_current'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_type \
  --features M \
  --seq_len $seq_len \
  --label_len $seq_len \
  --pred_len $pred_len \
  --enc_in $enc_in \
  --dec_in $enc_in \
  --c_out $enc_in \
  --des 'Exp' \
  --scale 1 \
  --plot_every 10 \
  --num_workers 8 \
  --train_epochs 100 \
  --max_grad_norm $max_grad_norm \
  --predictor $predictor \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --dropout $dropout \
  --patience $patience \
  --random_seed $random_seed \
  --random_drop_training $random_drop_training \
  --gradient_accumulation $gradient_accumulation \
  --pct_start 0.05 \
  --compile $torch_compile \
  --itr 1 --batch_size $batch_size --learning_rate 0.0001 >logs/LongForecasting/$random_seed'_'$model_name'_'$predictor'_'$data_alias_current'_'$seq_len'_'$pred_len'.log'

done
done
done
done