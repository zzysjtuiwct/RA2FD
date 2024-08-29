import torch,ipdb
import torch.nn.functional as F
from torch.nn import KLDivLoss, CrossEntropyLoss
import logging

logger = logging.getLogger(__name__)

def weight_masking(args, model, batch, tokenizer, teacher_model=None):
    input_ids_w_know, input_ids_wo_know, attention_mask_w_know, attention_mask_wo_know, lm_labels = batch

    loss_mask = (lm_labels != -100).unsqueeze(-1).expand(-1,-1,model.config.vocab_size)
    weight_mask = torch.ones_like(lm_labels, dtype=torch.float)
    
    weight_mask = weight_mask.unsqueeze(-1).expand(-1, -1, model.config.vocab_size)
    return loss_mask, weight_mask


def run_batch_generation(args, model, batch, tokenizer, batch_size: int, teacher_model=None):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids_w_know, input_ids_wo_know, attention_mask, lm_labels_w_know, lm_labels_wo_know = batch

    if args.knowledge_usage == "External":
        model_outputs = model(input_ids=input_ids_w_know, attention_mask=attention_mask, labels=lm_labels_w_know)
        loss = model_outputs.loss if lm_labels_wo_know.max() !=-100 else torch.tensor(999.9)
        lm_logits = model_outputs.logits
        track_loss = torch.tensor([0])
    elif args.knowledge_usage == "Paramete":
        model_outputs = model(input_ids=input_ids_wo_know, attention_mask=attention_mask, labels=lm_labels_wo_know)
        loss = model_outputs[0] if lm_labels_wo_know.max() !=-100 else torch.tensor(999.9)
        lm_logits = model_outputs[1]
        track_loss = torch.tensor([0])
    elif args.knowledge_usage == "Multilabel":
        # define parameters
        label_num = int(input_ids_wo_know.shape[0]/batch_size)
        
        # forward
        model_outputs = model(input_ids=input_ids_wo_know, attention_mask=attention_mask, labels=lm_labels_wo_know)

        shift_logits = model_outputs.logits[..., :-1, :].contiguous()
        shift_labels = lm_labels_wo_know[..., 1:].contiguous()
        none_ignore_mask = (shift_labels != -100)

        # main loss
        NLL_function = CrossEntropyLoss(reduction='none')
        NLL_loss = NLL_function(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).reshape(batch_size*label_num, -1)
        main_loss = (NLL_loss[::label_num,:].sum())/(none_ignore_mask[::label_num,:].sum())

        # likelihood margin loss
        label_mod = shift_labels * none_ignore_mask
        prob = F.log_softmax(shift_logits, dim=-1)
        log_prob = prob.gather(2, label_mod.unsqueeze(-1)).squeeze(-1)
        log_likelihoods = (log_prob * none_ignore_mask).sum(dim=1)/(none_ignore_mask.sum(dim=1))
        log_likelihoods_per_batch = log_likelihoods.reshape(batch_size, -1)

        # calculate difference
        diff_likelihoods = log_likelihoods_per_batch.unsqueeze(-1) - log_likelihoods_per_batch.unsqueeze(1)
        margin_diff_likelihoods = torch.triu(args.margin - diff_likelihoods, diagonal = 1)

        # diff_loss = args.margin - (log_likelihoods_per_batch[:, : label_num - 1] - log_likelihoods_per_batch[:, 1:])
        margin_loss = torch.max(margin_diff_likelihoods, torch.tensor(0.0, device='cuda:0')).sum()/(label_num*batch_size)

        # total loss
        loss = args.NLL_weight * main_loss + args.margin_weight * margin_loss
        track_loss = margin_loss
        lm_logits = model_outputs.logits

    return loss, lm_logits, track_loss, torch.tensor([])

def run_batch_generation_Encoder_Decoder(args, model, batch, tokenizer, batch_size: int, teacher_model=None):
    tag_token_id = tokenizer.convert_tokens_to_ids("<knowledge_tag>")
    end_token_id = tokenizer.convert_tokens_to_ids("<eos>")
    label_knowledge = batch[-1]['knowledge_texts']
    label_response = batch[-1]['label_texts']
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids_w_know, input_ids_wo_know, attention_mask_w_know, attention_mask_wo_know, lm_labels = batch
    if args.knowledge_usage == "External":
        model_outputs = model(input_ids=input_ids_w_know, attention_mask=attention_mask_w_know, labels=lm_labels)
        loss_mask, weight_mask = weight_masking(args, model, batch, tokenizer, teacher_model)
        
        lm_logits = model_outputs.logits
        weight_logits = lm_logits * weight_mask
        loss_fct = CrossEntropyLoss()

        loss = loss_fct(weight_logits.view(-1, model.config.vocab_size), lm_labels.view(-1))        
        track_loss = torch.tensor([0])
    elif args.knowledge_usage == "Paramete":
        model_outputs = model(input_ids=input_ids_wo_know, attention_mask=attention_mask_wo_know, labels=lm_labels)
        loss = model_outputs.loss
        lm_logits = model_outputs.logits
        track_loss = torch.tensor([0])
    elif args.knowledge_usage == "Multilabel":
        # define parameters
        label_num = int(input_ids_wo_know.shape[0]/batch_size)
        none_ignore_mask = (lm_labels != -100)

        # forward
        model_outputs = model(input_ids=input_ids_wo_know, attention_mask=attention_mask_wo_know, labels=lm_labels)

        # main loss
        NLL_function = CrossEntropyLoss(reduction='none')
        NLL_loss = NLL_function(model_outputs.logits.view(-1, model.config.vocab_size), lm_labels.view(-1)).reshape(batch_size*label_num, -1)
        main_loss = (NLL_loss[::label_num,:].sum())/(none_ignore_mask[::label_num,:].sum())

        # likelihood margin loss
        label_mod = lm_labels * none_ignore_mask
        prob = F.log_softmax(model_outputs.logits, dim=-1)
        log_prob = prob.gather(2, label_mod.unsqueeze(-1)).squeeze(-1)
        log_likelihoods = (log_prob * none_ignore_mask).sum(dim=1)/(none_ignore_mask.sum(dim=1))
        log_likelihoods_per_batch = log_likelihoods.reshape(batch_size, -1)

        # calculate difference
        diff_likelihoods = log_likelihoods_per_batch.unsqueeze(-1) - log_likelihoods_per_batch.unsqueeze(1)
        margin_diff_likelihoods = torch.triu(args.margin - diff_likelihoods, diagonal = 1)

        # diff_loss = args.margin - (log_likelihoods_per_batch[:, : label_num - 1] - log_likelihoods_per_batch[:, 1:])
        margin_loss = torch.max(margin_diff_likelihoods, torch.tensor(0.0, device='cuda:0')).sum()/(label_num*batch_size)

        # total loss
        loss = args.NLL_weight * main_loss + args.margin_weight * margin_loss
        track_loss = margin_loss
        lm_logits = model_outputs.logits

    return loss, lm_logits, track_loss, torch.tensor([])

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def run_batch_generation_sample(args, model, tokenizer, batch, dataset):
    special_tokens_ids = args.tokenizer.convert_tokens_to_ids(dataset.SPECIAL_TOKENS_VALUES)
    current_output = []

    example = batch[0]
    knowledge, history = example["knowledge"], example["history"]
    response_text = example["response_text"]
    dialog_id = example["dialog_id"]

    for i in range(args.max_length):
        instance, sequence = dataset.build_input_from_segments(
            knowledge, history, current_output, with_eos=False
        )

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        model_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        logits = model_outputs[0]

        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    logger.warning("Warning: model generating special token with probability 1! Breaking...")
                    break
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())
    
    return current_output, response_text, dialog_id

def run_batch_generation_sample_pure_Decoder(args, model, tokenizer, batch, dataset):
    current_output = []

    example = batch[0]
    knowledge, history = example["knowledge"], example["history"]
    response_text = example["response_text"]
    dialog_id = example["dialog_id"]

    instance, sequence = dataset.build_input_from_segments(
        knowledge if args.use_external_knowlegde else [], 
        history, 
        current_output, 
        with_eos=False
    )
    input_ids = torch.tensor(instance["input_ids"], device='cuda').unsqueeze(0)
    current_output = model.generate(input_ids=input_ids, num_beams=args.num_beams,
                                    min_length=args.min_length, max_length=args.max_length+input_ids.size(-1),
                                    eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
                                    pad_token_id=tokenizer.pad_token_id, do_sample=args.do_sample, 
                                    num_return_sequences=args.num_return_sequences, output_scores=True, return_dict_in_generate=True)
    return current_output.sequences[:, input_ids.size(-1):], response_text, dialog_id, example["knowledge_text"]

def run_batch_generation_sample_Encoder_Decoder(args, model, tokenizer, batch, dataset):
    """ Run batch generation during test time
        Responses are decoded using beam search + sampling
    """
    current_output = []

    example = batch[0]
    knowledge, history = example["knowledge"], example["history"]
    response_text = example["response_text"]
    dialog_id = example["dialog_id"]

    instance, sequence = dataset.build_input_from_segments(
        knowledge, history, current_output
    )

    if args.use_external_knowlegde:
        input_ids = torch.tensor(instance["input_ids_w_know"], device=args.device).unsqueeze(0)
    else:
        input_ids = torch.tensor(instance["input_ids_wo_know"], device=args.device).unsqueeze(0)
    
    current_output = model.generate(input_ids=input_ids, num_beams=args.num_beams,
                                    min_length=args.min_length, max_new_tokens=args.max_length,
                                    eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
                                    pad_token_id=tokenizer.pad_token_id, do_sample=args.do_sample, num_return_sequences=args.num_return_sequences)
    return current_output, response_text, dialog_id, example["knowledge_text"]


def run_batch_selection_train(args, model, batch, tokenizer, batch_size: int, teacher_model=None):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, mc_token_ids, attention_mask, mc_labels = batch
    model_outputs = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=mc_labels
    )
    mc_loss = model_outputs.loss
    mc_logits = model_outputs.logits
    return mc_loss, mc_logits, torch.tensor([0]), mc_labels


def run_batch_selection_eval(args, model, batch, tokenizer, batch_size: int, teacher_model=None):
    candidates_per_forward = args.max_candidates_per_forward_eval * (args.n_gpu if isinstance(model, torch.nn.DataParallel) else 1)
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, mc_token_ids, attention_mask, mc_labels = batch
    all_mc_logits = []
    for index in range(0, input_ids.size(1), candidates_per_forward):
        model_outputs = model(
            input_ids=input_ids[0, index:index+candidates_per_forward].unsqueeze(1),
            attention_mask=attention_mask[0, index:index+candidates_per_forward].unsqueeze(1),
        )
        mc_logits = model_outputs.logits
        all_mc_logits.append(mc_logits.detach())
    all_mc_logits = torch.cat(all_mc_logits, dim=0).unsqueeze(0)
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(all_mc_logits.squeeze(-1), mc_labels)
    return loss, torch.tensor([]), all_mc_logits, mc_labels


def run_batch_detection(args, model, batch):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, mc_token_ids, lm_labels, labels = batch
    model_outputs = model(
        input_ids=input_ids, token_type_ids=token_type_ids,
        mc_token_ids=mc_token_ids, labels=labels
    )
    cls_loss = model_outputs[0]
    lm_logits, cls_logits = model_outputs[1], model_outputs[2]
    return cls_loss, lm_logits, cls_logits, labels