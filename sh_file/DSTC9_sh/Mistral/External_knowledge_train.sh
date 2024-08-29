cd /root/RA2FD
export LANG=en_US.UTF-8
exp_name=External-Knowledge
knowledge_usage=External
lr=7e-6
epochs=10
dataset=DSTC9
pretrain_model=Lora-Mistral
para_adjust_name=${pretrain_model}-${knowledge_usage}-lr-${lr}
save_path=${dataset}/${exp_name}/$(date +%Y-%m-%d)/${para_adjust_name}
model_name_or_path=/root/download_model/Mistral-7B-v0.1
cuda_id=0
# Response generation
CUDA_VISIBLE_DEVICES=${cuda_id} python3 baseline.py \
    --exp_name ${save_path} \
    --model_name_or_path ${model_name_or_path} \
    --learning_rate ${lr} \
    --knowledge_usage ${knowledge_usage} \
    --use_external_in_evaluate \
    --dataroot data \
    --num_train_epochs ${epochs} \
    --DatasetClass ResponseGenerationDataset \
    --TrainFunc run_batch_generation \
    --EvalFunc run_batch_generation \
    --pure_decoder \
    --per_gpu_eval_batch_size 16 \
    --use_lora

results_list=("KF1_best" "BLEU_best")
for result in "${results_list[@]}";do
    rg_output_dir=runs/${save_path}/${result}/pred
    CUDA_VISIBLE_DEVICES=${cuda_id} python3 baseline.py \
        --generate runs/${save_path}/${result} \
        --DatasetClass ResponseGenerationEvalDataset \
        --rg_sampler run_batch_generation_sample_pure_Decoder \
        --output_file ${rg_output_dir}/rg.json \
        --use_lora \
        --model_name_or_path ${model_name_or_path} \
        --use_external_knowlegde

    python scripts/check_results.py --dataset test --dataroot data_eval/ --outfile ${rg_output_dir}/rg.json
    CUDA_VISIBLE_DEVICES=${cuda_id} python scripts/scores.py --dataset test --dataroot data_eval/ --outfile ${rg_output_dir}/rg.json --scorefile ${rg_output_dir}/score.json
    python scripts/Json2txt.py --file_path ${rg_output_dir}/score.json --exp_name ${exp_name} --output_path ${rg_output_dir}/score.txt
done

# generate multi-label
# please rename the orignal train folder as test and move it to runs/${save_path}/pred_${label_num}_label to let the code in the inference model
save_path=${dataset}/${exp_name}/$(date +%Y-%m-%d)/${para_adjust_name}
label_num=10
rg_output_dir=runs/${save_path}/pred_${label_num}_label
cuda_id=0
model_name_or_path=/root/download_model/Mistral-7B-v0.1
CUDA_VISIBLE_DEVICES=${cuda_id} python3 baseline.py \
    --generate runs/${save_path}/KF1_best \
    --DatasetClass ResponseGenerationEvalDataset \
    --rg_sampler run_batch_generation_sample_pure_Decoder \
    --output_file ${rg_output_dir}/rg.json \
    --use_external_knowlegde \
    --num_return_sequences ${label_num} \
    --dataroot runs/${save_path}/pred_${label_num}_label \
    --eval_dataset test \
    --labels_file runs/${save_path}/pred_${label_num}_label/test/labels.json \
    --num_beams_specify ${label_num} \
    --use_lora \
    --model_name_or_path ${model_name_or_path} \