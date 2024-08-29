cd /root/RA2FD
export LANG=en_US.UTF-8
exp_name=Parameter-Knowledge
knowledge_usage=Multilabel
lr=1e-4
epochs=10
dataset=DSTC9
pretrain_model=Lora-Inject-Mistral
label_num=5
margin=6
margin_weight=0.5
para_adjust_name=${pretrain_model}-${knowledge_usage}-lr-${lr}-Labelnum-${label_num}-margin-${margin}-margin_weight-${margin_weight}
save_path=${dataset}/${exp_name}/$(date +%Y-%m-%d)/${para_adjust_name}
model_name_or_path=runs/DSTC9/Inject-model/Mistral20/checkpoint-3760
cuda_id=0,1,2
# Response generation
# please move the original val dataset to the following dataroot
CUDA_VISIBLE_DEVICES=${cuda_id} python3 baseline.py \
    --exp_name ${save_path} \
    --model_name_or_path ${model_name_or_path} \
    --learning_rate ${lr} \
    --knowledge_usage ${knowledge_usage} \
    --dataroot runs/DSTC9/External-Knowledge/2024-05-28/Lora-Mistral-External-lr-7e-6/pred_10_label \
    --num_train_epochs ${epochs} \
    --DatasetClass ResponseGenerationDataset_Multilabel \
    --TrainFunc run_batch_generation \
    --EvalFunc run_batch_generation \
    --pure_decoder \
    --use_lora \
    --per_gpu_eval_batch_size 32 \
    --label_num ${label_num} \
    --margin ${margin} \
    --margin_weight ${margin_weight} \

results_list=("KF1_best" "BLEU_best")
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