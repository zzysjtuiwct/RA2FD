import logging
import os

import torch


logger = logging.getLogger(__name__)


def verify_args(args, parser):
    if args.eval_only:
        if not args.checkpoint:
            parser.error("--checkpoint is required when --eval_only is set.")
        if not args.params_file:
            logger.info("params_file is not set, using the params.json in checkpoint")
            args.params_file = os.path.join(args.checkpoint, "params.json")
        else:
            logger.info("Using params_file %s from command line", args.params_file)
    else:
        if not args.params_file:
            parser.error("--params_file is required during training")


def update_additional_params(params, args):
    if args.get("model_name_or_path"):
        params["model_name_or_path"] = args["model_name_or_path"]
    if args.get("warmup_steps"):
        params["warmup_steps"] = args["warmup_steps"]
    if args.get("learning_rate", False):
        params["learning_rate"] = args["learning_rate"]
    if args.get("num_train_epochs", False):
        params["num_train_epochs"] = args["num_train_epochs"]
    if args.get("seed", False):
        params["seed"] = args["seed"]
    if args.get("per_gpu_train_batch_size", False):
        params["per_gpu_train_batch_size"] = args["per_gpu_train_batch_size"]
    if args.get("gradient_accumulation_steps", False):
        params["gradient_accumulation_steps"] = args["gradient_accumulation_steps"]
    if args.get("knowledge_usage", False):
        params["knowledge_usage"] = args["knowledge_usage"]
    if args.get("NLL_weight", False):
        params["NLL_weight"] = args["NLL_weight"]
    if args.get("label_num", False):
        params["label_num"] = args["label_num"]
    if args.get("margin_weight", False):
        params["margin_weight"] = args["margin_weight"]
    if args.get("margin", False):
        params["margin"] = args["margin"]
    if args.get("per_gpu_eval_batch_size", False):
        params["per_gpu_eval_batch_size"] = args["per_gpu_eval_batch_size"]
    if args.get("use_external_in_evaluate", False):
        params["use_external_in_evaluate"] = args["use_external_in_evaluate"]
    if args.get("pure_decoder", False):
        params["pure_decoder"] = args["pure_decoder"]
    ''' used in model generation'''
    if args.get("num_beams_specify", False):
        params["num_beams_specify"] = args["num_beams_specify"]
    if args.get("eval_dataset", False):
        params["eval_dataset"] = args["eval_dataset"]
    if args.get("hard_negative", False):
        params["dataset_args"]["hard_negative"] = args["hard_negative"]
    
    if args.get("n_candidates", False):
        params["dataset_args"]["n_candidates"] = args["n_candidates"]

    '''above is additioanl parameters ''' 

    if args.get("dataroot"):
        params["dataset_args"]["dataroot"] = args["dataroot"]

    if args.get("knowledge_file"):
        params["dataset_args"]["knowledge_file"] = args["knowledge_file"]
    
    if args.get("negative_sample_method", ""):
        params["dataset_args"]["negative_sample_method"] = args["negative_sample_method"]
    
    if args.get("eval_all_snippets", False):
        params["dataset_args"]["eval_all_snippets"] = args["eval_all_snippets"]
    
    for key in ["history_max_tokens", "knowledge_max_tokens"]:
        if args.get(key, -1) > -1:
            params["dataset_args"][key] = args[key]


def set_attr_if_not_exists(args, name, value):
    if not hasattr(args, name):
        setattr(args, name, value)


def set_default_params(args):
    pass


def set_default_dataset_params(args):
    set_attr_if_not_exists(args, "n_candidates", 1)
    set_attr_if_not_exists(args, "eval_all_snippets", False)
    set_attr_if_not_exists(args, "negative_sample_method", "all")
    set_attr_if_not_exists(args, "history_max_utterances", 100000)
    set_attr_if_not_exists(args, "history_max_tokens", 128)
    set_attr_if_not_exists(args, "knowledge_max_tokens", 128)
