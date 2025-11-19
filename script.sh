# ==== 参数 ====
$seed = 0
$lr_u = 1e-5
$lr_w = 1e-5
$lr_lambda = 1e-2
$batch_size = 64
$micro_batch_size = 8
$val_batch_size = 1
$epoch = 1
$exp_name = "exp1_gpt2_demo"
$data_dir = "data_tiny"

# 统计 train_*.json 数量（Windows 用 Get-ChildItem）
$num_partitions = (Get-ChildItem "$data_dir/train_*.json").Count

$log_dir = "log_$exp_name"
$save_dir = "models/$exp_name"
$wandb_proj = "bilevel-optimization"
$alpha = 10

# 创建目录（Windows 下）
New-Item -ItemType Directory -Force -Path $log_dir | Out-Null
New-Item -ItemType Directory -Force -Path $save_dir | Out-Null

$exp_id = "${exp_name}_lr-u-$lr_u`_lr-w-$lr_w`_lr_lambda-$lr_lambda`_seed-$seed`_alpha-$alpha`_epoch-$epoch"

Write-Host "$(Get-Date): $exp_id"

# ==== 执行训练 ====
accelerate launch --config_file fsdp_config_gpt2.yaml python/main.py `
    --wandb_run_name $exp_id `
    --wandb_project $wandb_proj `
    --train-data "$data_dir/train_*.json" `
    --val-data "$data_dir/val.json" `
    --test-data "$data_dir/test.json" `
    --model "openai-community/gpt2" `
    --micro_batch_size $micro_batch_size `
    --global_batch_size $batch_size `
    --max-length 512 `
    --tokenizer-name "openai-community/gpt2" `
    --optimizer "name=adamw" `
    --init_lr 5e-4 `
    --minmax_init_lr_u $lr_u `
    --minmax_init_lr_w $lr_w `
    --minmax_init_lr_lambda $lr_lambda `
    --lr_scheduler "name=cosine, min_lr=0.0" `
    --validation_model_mode "train" `
    --minmax_validation_sampler "stochastic" `
    --sampler "stochastic" `
    --pseudo_random `
    --logging_conf_file "conf/common.log_conf" `
    --init_alpha $alpha `
    --tau 1 `
    --seed $seed `
    --epoch $epoch `
    --num_outer_iter 1 `
    --bf16 `
    --lisa `
    --model-type "gpt" `
    --use_wandb `
    --val_batch_size $val_batch_size `
    --eval_frequency 50 `
    --num_partitions $num_partitions `
    --response_loss_only `
    --save_dir $save_dir `
    1> "$log_dir/$exp_id.log" `
    2> "$log_dir/$exp_id.err"
