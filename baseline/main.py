import argparse,ipdb,time
import glob
import logging
import os
import random
import shutil
import json

from typing import Dict, List, Tuple
from argparse import Namespace

import numpy as np
import sklearn
import torch

from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    GPT2DoubleHeadsModel,
    GPT2LMHeadModel,
    PreTrainedModel,
    PreTrainedTokenizer,
    BartForConditionalGeneration,
    BlenderbotForConditionalGeneration,
    RobertaForMultipleChoice,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from .dataset import (
    ResponseGenerationDataset,
    KnowledgeSelectionDataset,
    KnowledgeTurnDetectionDataset,
    KnowledgeSelectionDataset_WoW,
    #''' DSTC9 '''
    ResponseGenerationDatasetEncoderDecoder,
    ResponseGenerationDatasetEncoderDecoder_neg,
    ResponseGenerationDatasetEncoderDecoder_MultiLabel,
    ResponseGenerationDataset_Multilabel,
    # '''wow'''
    ResponseGenerationDatasetEncoderDecoder_wow,
    ResponseGenerationDatasetEncoderDecoder_Eval_wow,
    ResponseGenerationDatasetEncoderDecoder_MultiLabel_wow,
    ResponseGenerationDatasetPureDecoder_wow,
    ResponseGenerationDatasetPureDecoder_Multilabel_wow,
    ResponseGenerationEvalDataset_wow,
    SPECIAL_TOKENS
)
from .models import GPT2ClsDoubleHeadsModel
from .utils.argument import (
    set_default_params,
    set_default_dataset_params,
    update_additional_params,
    verify_args
)
from .utils.model import (
    run_batch_detection,
    run_batch_generation,
    run_batch_selection_train,
    run_batch_selection_eval,
    run_batch_generation_Encoder_Decoder,
    run_batch_generation_sample_Encoder_Decoder
)
from .utils.data import write_selection_preds, write_detection_preds, write_selection_preds_wow
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from .utils.metrics import (
    BLEU, METEOR, ROUGE, KnowledgeF1
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def get_classes(args):
    if args.task.lower() == "generation":
        #return ResponseGenerationDatasetEncoderDecoder, BartForConditionalGeneration, run_batch_generation_Encoder_Decoder, run_batch_generation_Encoder_Decoder
        return eval(args.DatasetClass), eval(args.ModelClass), eval(args.TrainFunc), eval(args.EvalFunc)
    elif args.task.lower() == "selection":
        return KnowledgeSelectionDataset_WoW, RobertaForMultipleChoice, run_batch_selection_train, run_batch_selection_eval
    elif args.task.lower() == "detection":
        return KnowledgeTurnDetectionDataset, GPT2ClsDoubleHeadsModel, run_batch_detection, run_batch_detection
    else:
        raise ValueError("args.task not in ['generation', 'selection', 'detection'], got %s" % args.task)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn_train, run_batch_fn_eval, teacher_model =None) -> Tuple[int, float]:
    if args.local_rank in [-1, 0]:
        log_dir = os.path.join("runs", args.exp_name) if args.exp_name else None
        tb_writer = SummaryWriter(log_dir)
        args.output_dir = log_dir

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset)# if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=train_dataset.collate_fn
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = args.warmup_steps*t_total
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    global_step = 0
    model.zero_grad()
    train_iterator = trange(
        0, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # for reproducibility
    val_loss = float('inf')
    eval_score = 0
    hallu_score = 0

    for _ in train_iterator:
        start_epoch_time = time.time()
        local_steps = 0
        tr_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            loss, _, Track_loss, _ = run_batch_fn_train(args, model, batch, tokenizer, batch[-1]["batchsize"], teacher_model)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                local_steps += 1
                epoch_iterator.set_postfix(Loss=tr_loss/local_steps)

        end_epoch_time = time.time()
        elapsed_time = end_epoch_time - start_epoch_time
        print("Training Time cost of each epoch: ", elapsed_time)
        results = evaluate(args, eval_dataset, model, tokenizer, run_batch_fn_eval, teacher_model = teacher_model, desc=str(global_step))
        if args.local_rank in [-1, 0]:
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
            tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar("loss", tr_loss / local_steps, global_step)
            tb_writer.add_scalar("Track_loss", Track_loss.item(), global_step)

            if args.task == 'generation':
                if results['BLEU'] > eval_score:
                    logger.info(f"Find a better model performs in evaluate metrics: {results['BLEU']}")
                    eval_score = results['BLEU']
                    save_model(args, os.path.join(args.output_dir, 'BLEU_best'), model, tokenizer)
                else:
                    logger.info(f"The evaluate metrics: {results['BLEU']} is smaller than "
                                f"the biggest BLEU score {eval_score}, continue to train ... ")
                
                if results['KnowledgeF1'] > hallu_score:
                    logger.info(f"Find a better model performs in KnowledgeF1 metrics: {results['KnowledgeF1']}")
                    hallu_score = results['KnowledgeF1']
                    save_model(args, os.path.join(args.output_dir, 'KF1_best'), model, tokenizer)
                else:
                    logger.info(f"The evaluate metrics: {results['KnowledgeF1']} is smaller than "
                                f"the biggest KnowledgeF1 {hallu_score}, continue to train ... ")
            elif args.task == 'selection':
                if results['loss'] < val_loss:
                    logger.info(f"Find a better model performs in evaluate loss: {results['loss']}")
                    val_loss = results['loss']
                    save_model(args, os.path.join(args.output_dir, 'eval_loss_best'), model, tokenizer)
                else:
                    logger.info(f"The evaluate metrics: {results['loss']} is bigger than "
                                f"the smallest val loss {val_loss}, continue to train ... ")
                
                if results['accuracy'] > eval_score:
                    logger.info(f"Find a better model performs in accuracy: {results['accuracy']}")
                    eval_score = results['accuracy']
                    save_model(args, os.path.join(args.output_dir, 'accuracy_best'), model, tokenizer)
                else:
                    logger.info(f"The evaluate metrics: {results['accuracy']} is smaller than "
                                f"the biggest accuracy {eval_score}, continue to train ... ")

    if args.local_rank in [-1, 0]:
        tb_writer.flush()
        tb_writer.close()

    return global_step, tr_loss / local_steps

def save_model(args, output_dir, model, tokenizer):
    """ Save model, tokenizer, and params to the output dir """
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = (
                model.module if hasattr(model, "module") else model
            )
    logger.info("Saving model checkpoint to %s", output_dir)
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
        json.dump(args.params, jsonfile, indent=2, default=lambda x: str(x))

def evaluate(args, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn, teacher_model=None, desc="") -> Dict:
    if args.local_rank in [-1, 0]:
        eval_output_dir = args.output_dir
        os.makedirs(eval_output_dir, exist_ok=True)

    # eval_batch_size for selection must be 1 to handle variable number of candidates
    if args.task == "selection":
        args.eval_batch_size = 1
    else:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=eval_dataset.collate_fn
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and (args.task != "selection" or eval_dataset.args.eval_all_snippets):
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    data_infos = []
    all_preds = []
    all_labels = []
    All_inference_time = []
    metrics = [BLEU(), ROUGE(['rougeL']), KnowledgeF1()]
  
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            start_time = time.time()
            loss, lm_logits, mc_logits, mc_labels = run_batch_fn(args, model, batch, tokenizer, batch[-1]["batchsize"], teacher_model)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if args.task == "detection":
                mc_logits = mc_logits.sigmoid()
            if args.task in ["selection", "detection"]:
                data_infos.append(batch[-2])
                All_inference_time.append(elapsed_time)
            if args.task == "generation":
                ground_truth = batch[-1]['label_texts']
                used_knowledge_text = batch[-1]['knowledge_texts']
                batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
                input_ids_w_know, input_ids_wo_know, _, _, _ = batch

                input_ids = input_ids_w_know if args.use_external_in_evaluate else input_ids_wo_know

                if args.pure_decoder: # attention! left pad is need for pure decoder model to perform batch decode
                    for id in range(input_ids.shape[0]):
                        num_pad = torch.sum(input_ids[id] == tokenizer.pad_token_id).item()
                        input_ids[id] = torch.roll(input_ids[id], shifts=num_pad, dims=0)

                sampled_output_ids = model.generate(input_ids=input_ids, num_beams=args.num_beams,
                                min_length=args.min_length, max_length = args.max_length if not args.pure_decoder else args.max_length + input_ids.size(-1),
                                eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
                                pad_token_id=tokenizer.pad_token_id, do_sample=args.do_sample, num_return_sequences=1)
                
                sampled_output_text = tokenizer.batch_decode(sampled_output_ids if not args.pure_decoder else sampled_output_ids[:, input_ids.size(-1):],
                                                              skip_special_tokens=True)
                for metric in metrics:
                    for idx in range(len(sampled_output_text)):
                        metric.update((sampled_output_text[idx], used_knowledge_text[idx] if type(metric).__name__ == 'KnowledgeF1' else ground_truth[idx]))
        
            all_preds.append(mc_logits.detach().cpu().numpy())
            all_labels.append(mc_labels.detach().cpu().numpy())
            eval_loss += loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    if args.task.lower() == "generation":
        result = {}
        perplexity = torch.exp(torch.tensor(eval_loss)).item()
        total_score = 0
        for metric in metrics:
            total_score += metric.compute()
            result[metric.name()] = metric.compute()
        result.update({"perplexity": perplexity, "loss": eval_loss, "total_score":total_score})
    elif args.task.lower() == "selection":
        all_labels = np.array(all_labels).reshape(-1)
        all_pred_ids = np.array([np.argmax(logits) for logits in all_preds])
        accuracy = np.sum(all_pred_ids == all_labels) / len(all_labels)
        logger.info("Avg. # of candidates: %f", sum([len(arr[0]) for arr in all_preds]) / len(all_preds))
        result = {"loss": eval_loss, "accuracy": accuracy}
        if args.output_file:
            sorted_pred_ids = [np.argsort(logits.squeeze())[::-1] for logits in all_preds]
            if 'wow' in eval_dataset.dataroot:
                write_selection_preds_wow(eval_dataset.dataset_walker, args.output_file, data_infos, sorted_pred_ids, All_inference_time, topk=5)
            else:
                write_selection_preds(eval_dataset.dataset_walker, args.output_file, data_infos, sorted_pred_ids, All_inference_time, topk=5)
    elif args.task.lower() == "detection":
        all_labels = np.concatenate(all_labels)
        all_pred_ids = (np.concatenate(all_preds) > 0.5)
        accuracy = np.sum(all_pred_ids == all_labels) / len(all_labels)
        precision = sklearn.metrics.precision_score(all_labels, all_pred_ids)
        recall = sklearn.metrics.recall_score(all_labels, all_pred_ids)
        result = {"loss": eval_loss, "accuracy": accuracy, "precision": precision, "recall": recall}
        if args.output_file:
            write_detection_preds(eval_dataset.dataset_walker, args.output_file, data_infos, all_pred_ids)
    else:
        raise ValueError("args.task not in ['generation', 'selection', 'detection'], got %s" % args.task)

    if args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results %s *****" % desc)
            writer.write("***** Eval results %s *****\n" % desc)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--params_file", type=str, default="baseline/configs/generation/params.json", 
                        help="JSON configuration file")
    parser.add_argument("--eval_only", action="store_true",
                        help="Perform evaluation only")
    parser.add_argument("--checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument("--history_max_tokens", type=int, default=-1,
                        help="Maximum length in tokens for history, will override that value in config.")
    parser.add_argument("--knowledge_max_tokens", type=int, default=-1,
                        help="Maximum length in tokens for knowledge, will override that value in config.")
    parser.add_argument("--dataroot", type=str, default="data",
                        help="Path to dataset.")
    parser.add_argument("--knowledge_file", type=str, default="knowledge.json",
                        help="knowledge file name.")
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--no_labels", action="store_true",
                        help="Read a dataset without labels.json. This option is useful when running "
                             "knowledge-seeking turn detection on test dataset where labels.json is not available.")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="If set, the labels will be loaded not from the default path, but from this file instead."
                        "This option is useful to take the outputs from the previous task in the pipe-lined evaluation.")
    parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file.")
    parser.add_argument("--negative_sample_method", type=str, choices=["all", "mix", "oracle"],
                        default="", help="Negative sampling method for knowledge selection, will override the value in config.")
    parser.add_argument("--eval_all_snippets", action='store_true',
                        help="If set, the candidates to be selected would be all knowledge snippets, not sampled subset.")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Name of the experiment, checkpoints will be stored in runs/{exp_name}")
    parser.add_argument("--eval_desc", type=str, default="",
                        help="Optional description to be listed in eval_results.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--debug", type=int, default=0,
                        help="If set, will only use a small number (==debug) of data for training and test.")
    
    # additional parameters
    parser.add_argument("--model_name_or_path", type=str, help="model_name_or_path", default='gpt2')
    parser.add_argument("--warmup_steps", type=float, help="warmup_steps", default='0.2')
    parser.add_argument("--learning_rate", type=float, help="learning_rate", default=1e-5)
    parser.add_argument("--DatasetClass", type=str, default="ResponseGenerationDatasetEncoderDecoder")
    parser.add_argument("--ModelClass", type=str, default="BartForConditionalGeneration")
    parser.add_argument("--TrainFunc", type=str, default="run_batch_generation_Encoder_Decoder")
    parser.add_argument("--EvalFunc", type=str, default="run_batch_generation_Encoder_Decoder")
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=8)
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--knowledge_usage", type=str, 
                        choices=["External", 
                                 "Paramete", 
                                 "Multilabel"])
    parser.add_argument("--NLL_weight", type=float, default=1)
    parser.add_argument("--label_num", type=int, default=1)
    parser.add_argument("--margin", type=float, default=6)
    parser.add_argument("--margin_weight", type=float, default=0.5)
    parser.add_argument("--use_external_in_evaluate", action='store_true')
    parser.add_argument("--pure_decoder", action='store_true')
    parser.add_argument("--hard_negative", type=int, default=0)
    parser.add_argument("--n_candidates", type=int, default=2)
    parser.add_argument("--use_lora", action='store_true')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    verify_args(args, parser)

    # load args from params file and update the args Namespace
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)

        update_additional_params(params, args)
        args.update(params)
        args = Namespace(**args)
    
    args.params = params # used for saving checkpoints
    set_default_params(args)
    dataset_args = Namespace(**args.dataset_args)
    set_default_dataset_params(dataset_args)
    dataset_args.local_rank = args.local_rank
    dataset_args.task = args.task
    dataset_args.debug = args.debug
    dataset_args.label_num = args.label_num
    dataset_args.use_external_in_evaluate = args.use_external_in_evaluate

    args.n_gpu=1
    # Set seed
    set_seed(args)

    dataset_class, model_class, run_batch_fn_train, run_batch_fn_eval = get_classes(args)

    if args.eval_only:
        args.output_dir = args.checkpoint
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        model.to(args.device)
        logger.info("Training/evaluation parameters %s", args)
        if args.local_rank in [-1, 0]:
            eval_dataset = dataset_class(dataset_args, tokenizer, split_type=args.eval_dataset, labels=not args.no_labels, labels_file=args.labels_file)
            result = evaluate(args, eval_dataset, model, tokenizer, run_batch_fn_eval, desc=args.eval_desc or "val")
        return result
    else:
        if args.checkpoint is not None:
            model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        else:
            config = AutoConfig.from_pretrained(args.model_name_or_path)
            # set output_past to False for DataParallel to work during evaluation
            config.output_past = False
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            tokenizer.add_special_tokens(SPECIAL_TOKENS)
            tokenizer.model_max_length = min(1024, tokenizer.model_max_length)

            if 'bart' in args.model_name_or_path:
                model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)
                model.to(args.device)
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config, device_map="auto", torch_dtype =  torch.bfloat16)
            model.resize_token_embeddings(len(tokenizer))

            ### LoRA ####
            if args.use_lora:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, 
                    inference_mode=False, 
                    r=8, 
                    modules_to_save=["lm_head", "embed_tokens"], # We retrain these two modules
                    lora_alpha=32, 
                    lora_dropout=0.1,
                    target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
                )
                model = get_peft_model(model, peft_config)

            teacher_model = None
        
        logger.info("Training/evaluation parameters %s", args)
        train_dataset = dataset_class(dataset_args, tokenizer, split_type="train")
        eval_dataset = dataset_class(dataset_args, tokenizer, split_type=args.eval_dataset)
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer, run_batch_fn_train, run_batch_fn_eval, teacher_model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
