cd /root/RA2FD
export LANG=en_US.UTF-8
exp_name=Parameter-Knowledge
knowledge_usage=Paramete
lr=1e-4
epochs=10
dataset=DSTC9
pretrain_model=Lora-inject-Mistral
para_adjust_name=${pretrain_model}-${knowledge_usage}-lr-${lr}
save_path=${dataset}/${exp_name}/$(date +%Y-%m-%d)/${para_adjust_name}
model_name_or_path=runs/DSTC9/Inject-model/Mistral20/checkpoint-3760
cuda_id=0
# Response generation
CUDA_VISIBLE_DEVICES=${cuda_id} python3 baseline.py \
    --exp_name ${save_path} \
    --model_name_or_path ${model_name_or_path} \
    --learning_rate ${lr} \
    --knowledge_usage ${knowledge_usage} \
    --dataroot data \
    --num_train_epochs ${epochs} \
    --DatasetClass ResponseGenerationDataset \
    --TrainFunc run_batch_generation \
    --EvalFunc run_batch_generation \
    --pure_decoder \
    --use_lora \
    --per_gpu_eval_batch_size 16 \

results_list=("KF1_best" "total_score_best")
for result in "${results_list[@]}";do
    rg_output_dir=runs/${save_path}/${result}/pred
    CUDA_VISIBLE_DEVICES=${cuda_id} python3 baseline.py \
        --generate runs/${save_path}/${result} \
        --DatasetClass ResponseGenerationEvalDataset \
        --rg_sampler run_batch_generation_sample_pure_Decoder \
        --output_file ${rg_output_dir}/rg.json \
        --use_lora \
        --model_name_or_path ${model_name_or_path}

    python scripts/check_results.py --dataset test --dataroot data_eval/ --outfile ${rg_output_dir}/rg.json
    CUDA_VISIBLE_DEVICES=${cuda_id} python scripts/scores.py --dataset test --dataroot data_eval/ --outfile ${rg_output_dir}/rg.json --scorefile ${rg_output_dir}/score.json
    python scripts/Json2txt.py --file_path ${rg_output_dir}/score.json --exp_name ${exp_name} --output_path ${rg_output_dir}/score.txt
done