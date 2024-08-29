import argparse
import glob
import logging
import os
import random
import shutil
import json, ipdb, time

from typing import Dict, List, Tuple
from argparse import Namespace

import numpy as np
import sklearn
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AutoTokenizer, GPT2LMHeadModel, BartForConditionalGeneration, BlenderbotForConditionalGeneration, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from .dataset import (
    ResponseGenerationEvalDataset, 
    SPECIAL_TOKENS, 
    ResponseGenerationDatasetEncoderDecoder_Eval, 
    ResponseGenerationDatasetEncoderDecoder_Eval_wow,
    ResponseGenerationEvalDataset_wow
)
from .utils.argument import (
    set_default_params,
    set_default_dataset_params,
    update_additional_params,
    verify_args
)
from .utils.model import run_batch_generation_sample, run_batch_generation_sample_Encoder_Decoder, run_batch_generation_sample_pure_Decoder
from .utils.metrics import (
    UnigramMetric, NGramDiversity,
    CorpusNGramDiversity,
    BLEU, METEOR, ROUGE, KnowledgeF1
)
from .utils.data import write_generation_preds


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(args, eval_dataset, model, tokenizer, desc="") -> Dict:
    if args.local_rank in [-1, 0]:
        eval_output_dir = args.output_dir
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=1, # only support batch_size=1 for sampling right now
        collate_fn=eval_dataset.collate_fn
    )

    metrics = [
        UnigramMetric(metric="recall"),
        UnigramMetric(metric="precision"),
        NGramDiversity(n=1),
        NGramDiversity(n=2),
        NGramDiversity(n=3),
        NGramDiversity(n=4),
        CorpusNGramDiversity(n=1),
        CorpusNGramDiversity(n=2),
        CorpusNGramDiversity(n=3),
        CorpusNGramDiversity(n=4),
        BLEU(),
        METEOR(),
        ROUGE()
    ]
    hallu_metrics = [KnowledgeF1()]
    fluency_metrics = [BLEU(), ROUGE(['rougeL'])]
    All_metrics = hallu_metrics + fluency_metrics
    all_samples_scores = []

    args.tokenizer = tokenizer
    all_output_texts = []
    dialog_ids = []
    inference_time = []
    do_evaluate = False
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            start_time = time.time()
            sampled_output_ids, ground_truth, dialog_id, knowledge_text = eval(args.rg_sampler)(args, model, tokenizer, batch, 
                                                                                eval_dataset)
            sampled_output_text = [tokenizer.decode(_sampled_output_ids, skip_special_tokens=True) for 
                                   _sampled_output_ids in sampled_output_ids]
            end_time = time.time()
            elapsed_time = end_time - start_time

            dialog_ids.append(dialog_id)
            inference_time.append(elapsed_time)
        if ground_truth.strip() != "":
            do_evaluate = True
            for metric in metrics:
                metric.update((sampled_output_text[0], ground_truth))
            sample_result = {each_metric.name(): [] for each_metric in All_metrics}
            sample_result["total_score"] = []
            for sample in sampled_output_text:
                total_score = 0
                for each_metric in All_metrics:
                    each_metric.reset()
                    each_metric.update((sample, ground_truth)) if each_metric.name() != "KnowledgeF1" else each_metric.update((sample, knowledge_text))
                    each_score = each_metric.compute()
                    sample_result[each_metric.name()].append(each_score)
                    total_score+=each_score
                sample_result["total_score"].append(total_score)
            # sorted_texts, sorted_scores = zip(*sorted(zip(sampled_output_text, sample_result["total_score"]), key=lambda x: x[1], reverse=True))
            all_output_texts.append(sampled_output_text[0] if len(sampled_output_text) == 1 else list(sampled_output_text))
            all_samples_scores.append(sample_result["total_score"][0] if len(sample_result["total_score"]) == 1 else list(sample_result["total_score"]))

    if args.output_file:
        write_generation_preds(eval_dataset.dataset_walker, args.output_file, dialog_ids, all_output_texts, all_samples_scores, inference_time)

    result = dict()
    if do_evaluate and args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results %s *****" % desc)
            writer.write("***** Eval results %s *****\n" % desc)
            for metric in metrics:
                name = metric.name()
                score = metric.compute()
                if metric.is_single:
                    result[name] = score
                    logger.info("  %s = %s", name, str(score))
                    writer.write("%s = %s\n" % (name, str(score)))
                else:
                    for _name, _score in zip(name, score):
                        result[_name] = _score
                        logger.info("  %s = %s", _name, str(_score))
                        writer.write("%s = %s\n" % (_name, str(_score)))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument("--debug", type=int, default=0,
                        help="If set, will only use a small number (==debug) of data for training and test.")
    parser.add_argument('--generate', action='store_true')
    parser.add_argument("--generation_params_file", type=str, default="baseline/configs/generation/generation_params_encoder_decoder.json",
                        help="JSON configuration file for generation-related configurations.")
    parser.add_argument("--dataroot", type=str, default="data_eval",
                        help="Path to dataset, will override the path in config.")
    parser.add_argument("--eval_dataset", type=str, default="test",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--labels_file", type=str, default="data_eval/test/labels.json",
                        help="If set, the labels will be loaded not from the default path, but from this file instead."
                        "This option is useful to take the outputs from the previous task in the pipe-lined evaluation.")    
    parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file.")
    parser.add_argument("--eval_desc", type=str, default="",
                        help="Optional description to be listed in eval_results.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    '''add additional parameters to args 2023.11.2'''
    parser.add_argument("--ModelClass", type=str, default="BartForConditionalGeneration", help="Choose response generation Model")
    parser.add_argument("--DatasetClass", type=str, default="ResponseGenerationDatasetEncoderDecoder_Eval", help="Choose response generation Dataset")
    parser.add_argument("--rg_sampler", type=str, default="run_batch_generation_sample_Encoder_Decoder", help="Choose response generation Dataset class")
    parser.add_argument("--use_external_knowlegde", action='store_true')
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--num_beams_specify", type=int, default=5)
    parser.add_argument("--use_lora", action='store_true')
    parser.add_argument("--model_name_or_path", type=str)

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    # load args from params file and update the args Namespace
    args.params_file = os.path.join(args.checkpoint, "params.json")
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)
        update_additional_params(params, args)
        args.update(params)
        if len(args["generation_params_file"]) > 0:
            with open(args["generation_params_file"]) as fg:
                generation_params = json.load(fg)
            args.update(generation_params)
        args = Namespace(**args)
    
    args.params = params # used for saving checkpoints
    args.num_beams = args.num_beams_specify
    dataset_args = Namespace(**args.dataset_args)
    dataset_args.local_rank = args.local_rank
    dataset_args.task = args.task
    dataset_args.debug = args.debug

    # Setup CUDA, GPU & distributed training
    args.distributed = (args.local_rank != -1)
    if not args.distributed:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
    args.device = device

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.output_dir = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    if args.use_lora:
        ## lora ##
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.bfloat16)
        embedding_size = base_model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            base_model.resize_token_embeddings(len(tokenizer))
        lora_model = PeftModel.from_pretrained(base_model, args.checkpoint, torch_dtype=torch.bfloat16)
        model = lora_model.merge_and_unload()
        ## lora ##
    else:
        model = BartForConditionalGeneration.from_pretrained(args.checkpoint)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Generation parameters %s", args)
    
    # Evaluation
    result = {}
    if args.local_rank in [-1, 0]:
        eval_dataset = eval(args.DatasetClass)(dataset_args, tokenizer, split_type=args.eval_dataset, labels_file=args.labels_file)
        result = evaluate(args, eval_dataset, model, tokenizer, desc=args.eval_desc or "val")

    return result


if __name__ == "__main__":
    main()
