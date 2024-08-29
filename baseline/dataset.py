import os
import json
import random
import logging
import sys,ipdb

from itertools import chain

import torch

from tqdm import tqdm

from .utils.data import (
    pad_ids, truncate_sequences
)

from scripts.dataset_walker import DatasetWalker
from scripts.knowledge_reader import KnowledgeReader

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"],
}
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"]

class KnowledgeInjectionDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataroot, knowledge_file):
        self.knowledge_reader = KnowledgeReader(dataroot, knowledge_file)
        self.tokenizer = tokenizer
        
        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
        self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])

        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]

        self._prepare_knowledge()

    def _prepare_knowledge(self):
        self.knowledge_docs = self.knowledge_reader.get_doc_list()
        self.knowledge = []
        for snippet in tqdm(self.knowledge_docs, desc='prepare knowledge training sample...'):
            knowledge_Q, knowledge_A = self._knowledge_to_string(snippet["doc"],\
                                                                domain=snippet["domain"],\
                                                                name=snippet["entity_name"] or "")
            tokenized_Q = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(knowledge_Q))
            tokenized_Q = [self.bos] + tokenized_Q + [self.eos]
            tokenized_A = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(knowledge_A))
            tokenized_A = [self.bos] + tokenized_A + [self.eos]
            self.knowledge.append({"input_ids": tokenized_Q, "labels": tokenized_A, "attention_mask":[1]*len(tokenized_Q)})
    
    def _knowledge_to_string(self, doc, domain, name=""):
        join_str = " %s " % self.knowledge_sep_token
        Question = join_str.join([domain, name, doc["title"]])
        Answer = doc["body"]
        return Question, Answer

    def __getitem__(self, index):
        example = self.knowledge[index]
        return example
    
    def __len__(self):
        return len(self.knowledge)
    
    def collate_fn(self, batch):
        input_question = [example["Question"] for example in batch]
        input_answer = [example["Answer"] for example in batch]

        input_ids = torch.tensor(pad_ids(input_question, self.pad))
        attention_mask = 1 - (input_ids == self.pad).int()
        lm_labels = torch.tensor(pad_ids(input_answer, -100))

        return  input_ids, attention_mask, lm_labels

class KnowledgeInjectionDataset_puredecoder(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataroot, knowledge_file):
        self.knowledge_reader = KnowledgeReader(dataroot, knowledge_file)
        self.tokenizer = tokenizer
        
        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
        self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])

        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]

        self._prepare_knowledge()

    def _prepare_knowledge(self):
        self.knowledge_docs = self.knowledge_reader.get_doc_list()
        self.knowledge = []
        for snippet in tqdm(self.knowledge_docs, desc='prepare knowledge training sample...'):
            knowledge_Q, knowledge_A = self._knowledge_to_string(snippet["doc"],\
                                                                domain=snippet["domain"],\
                                                                name=snippet["entity_name"] or "")
            tokenized_Q = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(knowledge_Q))
            tokenized_A = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(knowledge_A))

            sequence = [[self.bos] + tokenized_Q]  + [tokenized_A + [self.eos]]

            lm_labels = ([-100] * sum(len(s) for s in sequence[:-1])) + sequence[-1]
            input_ids = list(chain(*sequence))
            self.knowledge.append({"input_ids": input_ids, "labels": lm_labels, "attention_mask": [1]*len(input_ids)})
    
    def _knowledge_to_string(self, doc, domain, name=""):
        join_str = " %s " % self.knowledge_sep_token
        Question = join_str.join([domain, name, doc["title"]])
        Answer = doc["body"]
        return Question, Answer

    def __getitem__(self, index):
        example = self.knowledge[index]
        return example
    
    def __len__(self):
        return len(self.knowledge)
    
    def collate_fn(self, batch):
        input_question = [example["Question"] for example in batch]
        input_answer = [example["Answer"] for example in batch]

        input_ids = torch.tensor(pad_ids(input_question, self.pad))
        attention_mask = 1 - (input_ids == self.pad).int()
        lm_labels = torch.tensor(pad_ids(input_answer, -100))

        return  input_ids, attention_mask, lm_labels

'''Dataset class used for WoW'''
class BaseDataset_wow(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type

        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
        self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
        self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"]
        )
        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]
        
        self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot=self.dataroot, labels_file=labels_file)
        self.dialogs = self._prepare_conversations()
        
        self._create_examples()
    
    def _prepare_conversations(self):
        logger.info("Tokenize and encode the dialog data")
        tokenized_dialogs = []
        for i, (log, label) in enumerate(tqdm(self.dataset_walker, disable=self.args.local_rank not in [-1, 0])): # only show progress bar in one process
            dialog = {}
            dialog["id"] = i
            dialog["log"] = log
            if label is not None:
                if "response" in label:
                    label["response_tokenized"] = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(label["response"])
                        ) if not isinstance(label["response"], list) else \
                    [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(res)) for res in label["response"][:1]]
                    if len(label["response_tokenized"])>1024:
                        print(label["response"])
            dialog["label"] = label
            tokenized_dialogs.append(dialog)
        return tokenized_dialogs
    
    def _create_examples(self):
        logger.info("Creating examples")
        self.examples = []
        for dialog in tqdm(self.dialogs, disable=self.args.local_rank not in [-1, 0]):
            if self.args.debug > 0 and len(self.examples) >= self.args.debug:
                break
            dialog_id = dialog["id"]
            label = dialog["label"]
            dialog = dialog["log"]
            if label is None:
                # This will only happen when running knowledge-seeking turn detection on test data
                # So we create dummy target here
                label = {"target": False}

            target = label["target"]

            if not target and self.args.task != "detection":
                # we only care about non-knowledge-seeking turns in turn detection task
                continue
            
            history = [
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(turn["text"]))
                for turn in dialog
            ]
            gt_resp = label.get("response", "")[:self.args.label_num] if isinstance(label.get("response", ""),list) else label.get("response", "")

            if isinstance(label.get("response", ""),list):
                resp_score = label.get("score", "")[:self.args.label_num]
                gt_resp, resp_score = zip(*sorted(zip(gt_resp, resp_score), key=lambda x: x[1], reverse=True))
                gt_resp, resp_score = list(gt_resp), list(resp_score)

            tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(gt_resp)) if not isinstance(gt_resp, list) else \
            [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(res)) for res in gt_resp]

            # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
            truncated_history = history[-self.args.history_max_utterances:]

            # perform token-level truncation of history from the left 
            truncated_history = truncate_sequences(truncated_history, self.args.history_max_tokens)

            if target:
                if "knowledge" not in label:
                    # when the labels.json is from knowledge-seeking turn detection,
                    # there will be no ground truth knowledge
                    # so we just use a dummy snippet here
                    if not self.args.eval_all_snippets:
                        raise ValueError("eval_all_snippets is required to be true when taking output from knowledge-seeking turn detection")
                    label["knowledge"] = [self.knowledge_docs[0]]

                knowledge = label["knowledge"][0] if isinstance(label["knowledge"],list) else label["knowledge"]
                knowledge_candidates = label['knowledge_candidate'] if 'knowledge_candidate' in label else None
                
                used_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(knowledge))
                used_knowledge = used_knowledge[:self.args.knowledge_max_tokens]
            else:
                knowledge_candidates = None
                used_knowledge = []

            self.examples.append({
                "history": truncated_history,
                "knowledge": used_knowledge,
                "knowledge_text": knowledge,
                "candidates": knowledge_candidates,
                "response": tokenized_gt_resp,
                "response_text": gt_resp,
                "label": label,
                "knowledge_seeking": target,
                "dialog_id": dialog_id,
                "hard_candidates": label['hard_candidate'] if 'hard_candidate' in label else None
            })

    def build_input_from_segments(self, knowledge, history, response, with_eos=True):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}

        sequence = [[self.bos] + knowledge] + history + [response + ([self.eos] if with_eos else [])]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]

        return instance, sequence

    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.examples)

class ResponseGenerationDatasetEncoderDecoder_wow(BaseDataset_wow):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationDatasetEncoderDecoder_wow, self).__init__(args, tokenizer, split_type, labels, labels_file)
    
    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(
            example["knowledge"],
            example["history"],
            example["response"]
        )
        instance["response_text"] = example["response_text"]
        instance["knowledge_text"] = example["knowledge_text"]
        return instance

    def collate_fn(self, batch):
        input_ids_w_know = [ins["input_ids_w_know"] for ins in batch]
        input_ids_wo_know = [ins["input_ids_wo_know"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]
        label_texts = [ins["response_text"] for ins in batch]
        knowledge_texts = [ins["knowledge_text"] for ins in batch]

        input_ids_w_know = torch.tensor(pad_ids(input_ids_w_know, self.pad))
        input_ids_wo_know = torch.tensor(pad_ids(input_ids_wo_know, self.pad))

        attention_mask_w_know = 1 - (input_ids_w_know == self.pad).int()
        attention_mask_wo_know = 1 - (input_ids_wo_know == self.pad).int()

        lm_labels = torch.tensor(pad_ids(lm_labels, -100))

        return input_ids_w_know, input_ids_wo_know, \
            attention_mask_w_know, attention_mask_wo_know, \
            lm_labels, {'knowledge_texts':knowledge_texts, 'label_texts':label_texts, "batchsize": len(batch)}
    
    def build_input_from_segments(self, knowledge, history, response, with_eos=True):
        instance = {}
        
        sequence = [knowledge] + history + [response]
        sequence_with_speaker = [# speaker 2 (user)
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        history = list(chain(*sequence_with_speaker[:-1]))

        sequence_w_Knowledge = [[self.bos]] + [sequence[0]] + [[self.knowledge_tag]] + [history] + [[self.eos]]
        sequence_wo_Knowledge = [[self.bos]] + [[]] + [[self.knowledge_tag]] + [history] + [[self.eos]]

        instance["input_ids_w_know"] = list(chain(*sequence_w_Knowledge))
        instance["input_ids_wo_know"] = list(chain(*sequence_wo_Knowledge))
        instance["lm_labels"] = [self.bos] + sequence_with_speaker[-1] + [self.eos]

        return instance, sequence

class ResponseGenerationDatasetEncoderDecoder_Eval_wow(ResponseGenerationDatasetEncoderDecoder_wow):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationDatasetEncoderDecoder_Eval_wow, self).__init__(args, tokenizer, split_type, labels, labels_file)
    
    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch

class ResponseGenerationDatasetEncoderDecoder_MultiLabel_wow(BaseDataset_wow):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationDatasetEncoderDecoder_MultiLabel_wow, self).__init__(args, tokenizer, split_type, labels, labels_file)
    
    def __getitem__(self, index):
        example = self.examples[index]
        instance_list = [self.build_input_from_segments(
            example["knowledge"],
            example["history"],
            example["response"]
        )[0]] if not isinstance(example["response_text"], list) else [self.build_input_from_segments(
            example["knowledge"],
            example["history"],
            res)[0] for res in example["response"]]
        
        instance = dict()
        for d in instance_list:
            for key, value in d.items():
                instance.setdefault(key, []).append(value)

        instance["response_text"] = example["response_text"]
        instance["knowledge_text"] = example["knowledge_text"]
        return instance

    def collate_fn(self, batch):
        input_ids_w_know = [ids for ins in batch for ids in ins["input_ids_w_know"]]
        input_ids_wo_know = [ids for ins in batch for ids in ins["input_ids_wo_know"]]
        lm_labels = [ids for ins in batch for ids in ins["lm_labels"]]
        label_texts = [ids for ins in batch for ids in ins["response_text"]]
        knowledge_texts = [ins["knowledge_text"] for ins in batch]

        input_ids_w_know = torch.tensor(pad_ids(input_ids_w_know, self.pad))
        input_ids_wo_know = torch.tensor(pad_ids(input_ids_wo_know, self.pad))

        attention_mask_w_know = 1 - (input_ids_w_know == self.pad).int()
        attention_mask_wo_know = 1 - (input_ids_wo_know == self.pad).int()

        lm_labels = torch.tensor(pad_ids(lm_labels, -100))

        return input_ids_w_know, input_ids_wo_know, \
            attention_mask_w_know, attention_mask_wo_know, \
            lm_labels, {'knowledge_texts':knowledge_texts, 'label_texts':label_texts, "batchsize": len(batch)}
    
    def build_input_from_segments(self, knowledge, history, response, with_eos=True):
        instance = {}
        
        sequence = [knowledge] + history + [response]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        history = list(chain(*sequence_with_speaker[:-1]))

        sequence_w_Knowledge = [[self.bos]] + [sequence[0]] + [[self.knowledge_tag]] + [history] + [[self.eos]]
        sequence_wo_Knowledge = [[self.bos]] + [[]] + [[self.knowledge_tag]] + [history] + [[self.eos]]

        instance["input_ids_w_know"] = list(chain(*sequence_w_Knowledge))
        instance["input_ids_wo_know"] = list(chain(*sequence_wo_Knowledge))
        instance["lm_labels"] = [self.bos] + sequence_with_speaker[-1] + [self.eos]

        return instance, sequence

class ResponseGenerationDatasetPureDecoder_wow(BaseDataset_wow):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationDatasetPureDecoder_wow, self).__init__(args, tokenizer, split_type, labels, labels_file)
    
    def __getitem__(self, index):
        example = self.examples[index]
        rt_instance = dict()
        rt_instance["response_text"] = example["response_text"]
        rt_instance["knowledge_text"] = example["knowledge_text"]

        instance, _ = self.build_input_from_segments(
            example["knowledge"], 
            example["history"], 
            example["response"] if self.split_type == 'train' else [],
            with_eos=True if self.split_type == 'train' else False
        )
        rt_instance["input_ids_w_know"] = instance["input_ids"]
        rt_instance["lm_labels_w_know"] = instance["lm_labels"]

        instance, _ = self.build_input_from_segments(
            [], 
            example["history"], 
            example["response"] if self.split_type == 'train' else [],
            with_eos=True if self.split_type == 'train' else False
        )
        rt_instance["input_ids_wo_know"] = instance["input_ids"]
        rt_instance["lm_labels_wo_know"] = instance["lm_labels"]
        return rt_instance

    def collate_fn(self, batch):
        input_ids_w_know = [ins["input_ids_w_know"] for ins in batch]
        input_ids_wo_know = [ins["input_ids_wo_know"] for ins in batch]

        lm_labels_w_know = [ins["lm_labels_w_know"] for ins in batch]
        lm_labels_wo_know = [ins["lm_labels_wo_know"] for ins in batch]

        input_ids_w_know = torch.tensor(pad_ids(input_ids_w_know, self.pad))
        input_ids_wo_know = torch.tensor(pad_ids(input_ids_wo_know, self.pad))
        
        attention_mask_w_know = 1 - (input_ids_w_know == self.pad).int()
        attention_mask_wo_know = 1 - (input_ids_wo_know == self.pad).int()

        lm_labels_w_know = torch.tensor(pad_ids(lm_labels_w_know, -100))
        lm_labels_wo_know = torch.tensor(pad_ids(lm_labels_wo_know, -100))

        label_texts = [ins["response_text"] for ins in batch]
        knowledge_texts = [ins["knowledge_text"] for ins in batch]
        # we only want to use input_ids_wo_know and in order to match the return number of ResponseGenerationDatasetEncoderDecoder, we only return attention_mask_wo_know
        if self.args.use_external_in_evaluate:
            return input_ids_w_know, input_ids_wo_know, attention_mask_w_know,\
                lm_labels_w_know, lm_labels_wo_know,\
                {'knowledge_texts':knowledge_texts, 'label_texts':label_texts, "batchsize": len(batch)}
        else:
            return input_ids_w_know, input_ids_wo_know, attention_mask_wo_know,\
                lm_labels_w_know, lm_labels_wo_know,\
                {'knowledge_texts':knowledge_texts, 'label_texts':label_texts, "batchsize": len(batch)}

class ResponseGenerationDatasetPureDecoder_Multilabel_wow(BaseDataset_wow):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationDatasetPureDecoder_Multilabel_wow, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        rt_instance = {"input_ids_w_know":[],
                       "lm_labels_w_know":[],
                       "input_ids_wo_know":[],
                       "lm_labels_wo_know":[],
                       "response_text": example["response_text"],
                       "knowledge_text": example["knowledge_text"]
                       }
        response_list= example["response"] if self.split_type == 'train' else [[]]

        instance_list = [self.build_input_from_segments(
            example["knowledge"],
            example["history"],
            res,
            with_eos=True if self.split_type == 'train' else False)[0] for res in response_list
        ]

        for instance in instance_list:
            rt_instance["input_ids_w_know"].append(instance['input_ids'])
            rt_instance["lm_labels_w_know"].append(instance['lm_labels'])


        instance_list = [self.build_input_from_segments(
            [],
            example["history"],
            res,
            with_eos=True if self.split_type == 'train' else False)[0] for res in response_list
        ]
        for instance in instance_list:
            rt_instance["input_ids_wo_know"].append(instance["input_ids"])
            rt_instance["lm_labels_wo_know"].append(instance["lm_labels"])
        return rt_instance

    def collate_fn(self, batch):
        input_ids_w_know = [ids for ins in batch for ids in ins["input_ids_w_know"]]
        input_ids_wo_know = [ids for ins in batch for ids in ins["input_ids_wo_know"]]

        lm_labels_w_know = [ids for ins in batch for ids in ins["lm_labels_w_know"]]
        lm_labels_wo_know = [ids for ins in batch for ids in ins["lm_labels_wo_know"]]

        input_ids_w_know = torch.tensor(pad_ids(input_ids_w_know, self.pad))
        input_ids_wo_know = torch.tensor(pad_ids(input_ids_wo_know, self.pad))
        
        attention_mask_w_know = 1 - (input_ids_w_know == self.pad).int()
        attention_mask_wo_know = 1 - (input_ids_wo_know == self.pad).int()

        lm_labels_w_know = torch.tensor(pad_ids(lm_labels_w_know, -100))
        lm_labels_wo_know = torch.tensor(pad_ids(lm_labels_wo_know, -100))

        label_texts = [ids for ins in batch for ids in ins["response_text"]]
        knowledge_texts = [ins["knowledge_text"] for ins in batch]
        # we only want to use input_ids_wo_know and in order to match the return number of ResponseGenerationDatasetEncoderDecoder, we only return attention_mask_wo_know
        if self.args.use_external_in_evaluate:
            return input_ids_w_know, input_ids_wo_know, attention_mask_w_know,\
                lm_labels_w_know, lm_labels_wo_know,\
                {'knowledge_texts':knowledge_texts, 'label_texts':label_texts, "batchsize": len(batch)}
        else:
            return input_ids_w_know, input_ids_wo_know, attention_mask_wo_know,\
                lm_labels_w_know, lm_labels_wo_know,\
                {'knowledge_texts':knowledge_texts, 'label_texts':label_texts, "batchsize": len(batch)}

class ResponseGenerationEvalDataset_wow(BaseDataset_wow):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationEvalDataset_wow, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch

class KnowledgeSelectionDataset_WoW(BaseDataset_wow):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeSelectionDataset_WoW, self).__init__(args, tokenizer, split_type, labels, labels_file)
        if self.args.negative_sample_method not in ["all", "mix", "oracle"]:
            raise ValueError("negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)

    def _knowledge_to_string(self, doc, name=""):
        join_str = " %s " % self.knowledge_sep_token
        return join_str.join([name, doc["title"], doc["body"]])

    def __getitem__(self, index):
        example = self.examples[index]

        this_inst = {
            "dialog_id": example["dialog_id"],
            "input_ids": [],
            "token_type_ids": [],
            "mc_token_ids": []
        }
        
        candidates = example["candidates"] # only one choice, not like DSTC9

        if self.split_type == "train":
            if self.args.hard_negative and len(example['hard_candidates']) != 0:
                hard_candidates = example['hard_candidates'].copy()
                copy_candidates = candidates.copy()

                hard_candidates.remove(example["knowledge_text"])
                copy_candidates.remove(example["knowledge_text"])

                new_cand = [example["knowledge_text"]]
                for cand in hard_candidates[:self.args.hard_negative]:
                    new_cand.append(cand)
                    copy_candidates.remove(cand)
                
                other_cands = random.sample(copy_candidates, k=self.args.n_candidates-len(new_cand))
                new_cand.extend(other_cands)
                random.shuffle(new_cand)
                candidates = new_cand
            else:
                candidates = self._shrink_label_cands(example["knowledge_text"], candidates)
                
        label_idx = candidates.index(example["knowledge_text"])

        this_inst["label_idx"] = label_idx
        tokenize_candidates = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(cand))[:self.args.knowledge_max_tokens] 
                               for cand in candidates]
        for cand in tokenize_candidates:
            instance, _ = self.build_input_from_segments(
                cand,
                example["history"]
            )
            this_inst["input_ids"].append(instance["input_ids"])
            this_inst["token_type_ids"].append(instance["token_type_ids"])
            this_inst["mc_token_ids"].append(instance["mc_token_ids"])

        return this_inst

    def build_input_from_segments(self, knowledge, history):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        instance = {}

        sequence = [[self.bos]] + history
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker + [[self.knowledge_tag] + knowledge + [self.eos]]

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence[:-1]) for _ in s] + [self.knowledge_tag for _ in sequence[-1]]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        return instance, sequence
    
    def _shrink_label_cands(self, label, candidates):
        shrunk_label_cands = candidates.copy()
        shrunk_label_cands.remove(label)
        shrunk_label_cands = random.sample(shrunk_label_cands, k=self.args.n_candidates-1)
        shrunk_label_cands.append(label)
        random.shuffle(shrunk_label_cands)

        return shrunk_label_cands

    def collate_fn(self, batch):
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
        mc_token_ids = [id for ins in batch for id in ins["mc_token_ids"]]
        label_idx = [ins["label_idx"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
        }

        batch_size = len(batch)
        n_candidates = len(batch[0]["input_ids"])

        input_ids = torch.tensor(
            pad_ids(input_ids, self.pad)
        ).view(batch_size, n_candidates, -1)
        
        attention_mask = 1 - (input_ids == self.pad).int()

        token_type_ids = torch.tensor(
            pad_ids(token_type_ids, self.pad)
        ).view(batch_size, n_candidates, -1)

        lm_labels = torch.full_like(input_ids, -100)
        mc_token_ids = torch.tensor(mc_token_ids).view(batch_size, n_candidates)
        label_idx = torch.tensor(label_idx)

        return input_ids, token_type_ids, mc_token_ids, attention_mask, label_idx, data_info, {"batchsize": len(batch)}

'''Dataset class used for DSTC9'''

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type

        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
        self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
        self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"]
        )
        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]

        self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot=self.dataroot, labels_file=labels_file)
        self.dialogs = self._prepare_conversations()

        self.knowledge_reader = KnowledgeReader(self.dataroot, args.knowledge_file)
        self.knowledge_text, self.snippets = self._prepare_knowledge()

        self._create_examples()

    def _prepare_conversations(self):
        logger.info("Tokenize and encode the dialog data")
        tokenized_dialogs = []
        for i, (log, label) in enumerate(tqdm(self.dataset_walker, disable=self.args.local_rank not in [-1, 0])): # only show progress bar in one process
            dialog = {}
            dialog["id"] = i
            dialog["log"] = log
            if label is not None:
                if "response" in label:
                    label["response_tokenized"] = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(label["response"])
                        ) if not isinstance(label["response"], list) else \
                    [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(res)) for res in label["response"][:1]]
            dialog["label"] = label
            tokenized_dialogs.append(dialog)
        return tokenized_dialogs
    
    def _prepare_knowledge(self):
        knowledge = self.knowledge_reader.knowledge
        self.knowledge_docs = self.knowledge_reader.get_doc_list()

        tokenized_snippets = dict()
        knowledge_text = dict()
        for snippet in self.knowledge_docs:
            key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]) or "", snippet["doc_id"])
            knowledge = self._knowledge_to_string(snippet["doc"], name=snippet["entity_name"] or "")
            tokenized_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(knowledge))
            tokenized_snippets[key] = tokenized_knowledge[:self.args.knowledge_max_tokens]
            knowledge_text[key] = knowledge
        return knowledge_text, tokenized_snippets

    def _knowledge_to_string(self, doc, name=""):
        return doc["body"]

    def _create_examples(self):
        logger.info("Creating examples")
        self.examples = []
        for dialog in tqdm(self.dialogs, disable=self.args.local_rank not in [-1, 0]):
            if self.args.debug > 0 and len(self.examples) >= self.args.debug:
                break
            dialog_id = dialog["id"]
            label = dialog["label"]
            dialog = dialog["log"]
            if label is None:
                # This will only happen when running knowledge-seeking turn detection on test data
                # So we create dummy target here
                label = {"target": False}

            target = label["target"]

            if not target and self.args.task != "detection":
                # we only care about non-knowledge-seeking turns in turn detection task
                continue
            
            history = [
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(turn["text"]))
                for turn in dialog
            ]
            gt_resp = label.get("response", "")[:self.args.label_num] if isinstance(label.get("response", ""),list) else label.get("response", "")

            if isinstance(label.get("response", ""),list):
                resp_score = label.get("score", "")[:self.args.label_num]
                gt_resp, resp_score = zip(*sorted(zip(gt_resp, resp_score), key=lambda x: x[1], reverse=True))
                gt_resp, resp_score = list(gt_resp), list(resp_score)

            tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(gt_resp)) if not isinstance(gt_resp, list) else \
            [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(res)) for res in gt_resp]

            # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
            truncated_history = history[-self.args.history_max_utterances:]

            # perform token-level truncation of history from the left 
            truncated_history = truncate_sequences(truncated_history, self.args.history_max_tokens)

            if target:
                if "knowledge" not in label:
                    # when the labels.json is from knowledge-seeking turn detection,
                    # there will be no ground truth knowledge
                    # so we just use a dummy snippet here
                    if not self.args.eval_all_snippets:
                        raise ValueError("eval_all_snippets is required to be true when taking output from knowledge-seeking turn detection")
                    label["knowledge"] = [self.knowledge_docs[0]]

                knowledge = label["knowledge"][0]
                knowledge_key = "{}__{}__{}".format(knowledge["domain"], knowledge["entity_id"], knowledge["doc_id"])
                # find snippets with same entity as candidates
                prefix = "{}__{}".format(knowledge["domain"], knowledge["entity_id"])
                knowledge_candidates = [
                    cand
                    for cand in self.snippets.keys() 
                    if "__".join(cand.split("__")[:-1]) == prefix
                ]
                if self.split_type == "train" and self.args.negative_sample_method == "oracle":
                    # if there's not enough candidates during training, we just skip this example
                    if len(knowledge_candidates) < self.args.n_candidates:
                        continue
                used_knowledge = self.snippets[knowledge_key]
                used_knowledge = used_knowledge[:self.args.knowledge_max_tokens]
            else:
                knowledge_candidates = None
                used_knowledge = []

            self.examples.append({
                "history": truncated_history,
                "knowledge": used_knowledge,
                "knowledge_text": self.knowledge_text[knowledge_key],
                "candidates": knowledge_candidates,
                "response": tokenized_gt_resp,
                "response_text": gt_resp,
                "label": label,
                "knowledge_seeking": target,
                "dialog_id": dialog_id
            })

    def build_input_from_segments(self, knowledge, history, response, with_eos=True):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}

        sequence = [[self.bos] + knowledge] + history + [response + ([self.eos] if with_eos else [])]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]

        return instance, sequence
                
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.examples)

class ResponseGenerationDatasetEncoderDecoder(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationDatasetEncoderDecoder, self).__init__(args, tokenizer, split_type, labels, labels_file)
    
    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(
            example["knowledge"],
            example["history"],
            example["response"]
        )
        instance["response_text"] = example["response_text"]
        instance["knowledge_text"] = example["knowledge_text"]
        return instance

    def collate_fn(self, batch):
        input_ids_w_know = [ins["input_ids_w_know"] for ins in batch]
        input_ids_wo_know = [ins["input_ids_wo_know"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]
        label_texts = [ins["response_text"] for ins in batch]
        knowledge_texts = [ins["knowledge_text"] for ins in batch]

        input_ids_w_know = torch.tensor(pad_ids(input_ids_w_know, self.pad))
        input_ids_wo_know = torch.tensor(pad_ids(input_ids_wo_know, self.pad))

        attention_mask_w_know = 1 - (input_ids_w_know == self.pad).int()
        attention_mask_wo_know = 1 - (input_ids_wo_know == self.pad).int()

        lm_labels = torch.tensor(pad_ids(lm_labels, -100))

        return input_ids_w_know, input_ids_wo_know, \
            attention_mask_w_know, attention_mask_wo_know, \
            lm_labels, {'knowledge_texts':knowledge_texts, 'label_texts':label_texts, "batchsize": len(batch)}
    
    def build_input_from_segments(self, knowledge, history, response, with_eos=True):
        instance = {}
        
        sequence = [knowledge] + history + [response]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        history = list(chain(*sequence_with_speaker[:-1]))

        sequence_w_Knowledge = [[self.bos]] + [sequence[0]] + [[self.knowledge_tag]] + [history] + [[self.eos]]
        sequence_wo_Knowledge = [[self.bos]] + [[]] + [[self.knowledge_tag]] + [history] + [[self.eos]]

        instance["input_ids_w_know"] = list(chain(*sequence_w_Knowledge))
        instance["input_ids_wo_know"] = list(chain(*sequence_wo_Knowledge))
        instance["lm_labels"] = [self.bos] + sequence_with_speaker[-1] + [self.eos]

        return instance, sequence

class ResponseGenerationDatasetEncoderDecoder_Eval(ResponseGenerationDatasetEncoderDecoder):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationDatasetEncoderDecoder_Eval, self).__init__(args, tokenizer, split_type, labels, labels_file)
    
    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch

class ResponseGenerationDatasetEncoderDecoder_MultiLabel(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationDatasetEncoderDecoder_MultiLabel, self).__init__(args, tokenizer, split_type, labels, labels_file)
    
    def __getitem__(self, index):
        example = self.examples[index]
        instance_list = [self.build_input_from_segments(
            example["knowledge"],
            example["history"],
            example["response"]
        )[0]] if not isinstance(example["response_text"], list) else [self.build_input_from_segments(
            example["knowledge"],
            example["history"],
            res)[0] for res in example["response"]]
        
        instance = dict()
        for d in instance_list:
            for key, value in d.items():
                instance.setdefault(key, []).append(value)

        instance["response_text"] = example["response_text"]
        instance["knowledge_text"] = example["knowledge_text"]
        return instance

    def collate_fn(self, batch):
        input_ids_w_know = [ids for ins in batch for ids in ins["input_ids_w_know"]]
        input_ids_wo_know = [ids for ins in batch for ids in ins["input_ids_wo_know"]]
        lm_labels = [ids for ins in batch for ids in ins["lm_labels"]]
        label_texts = [ids for ins in batch for ids in ins["response_text"]]
        knowledge_texts = [ins["knowledge_text"] for ins in batch]

        input_ids_w_know = torch.tensor(pad_ids(input_ids_w_know, self.pad))
        input_ids_wo_know = torch.tensor(pad_ids(input_ids_wo_know, self.pad))

        attention_mask_w_know = 1 - (input_ids_w_know == self.pad).int()
        attention_mask_wo_know = 1 - (input_ids_wo_know == self.pad).int()

        lm_labels = torch.tensor(pad_ids(lm_labels, -100))

        return input_ids_w_know, input_ids_wo_know, \
            attention_mask_w_know, attention_mask_wo_know, \
            lm_labels, {'knowledge_texts':knowledge_texts, 'label_texts':label_texts, "batchsize": len(batch)}
    
    def build_input_from_segments(self, knowledge, history, response, with_eos=True):
        instance = {}
        
        sequence = [knowledge] + history + [response]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        history = list(chain(*sequence_with_speaker[:-1]))

        sequence_w_Knowledge = [[self.bos]] + [sequence[0]] + [[self.knowledge_tag]] + [history] + [[self.eos]]
        sequence_wo_Knowledge = [[self.bos]] + [[]] + [[self.knowledge_tag]] + [history] + [[self.eos]]

        instance["input_ids_w_know"] = list(chain(*sequence_w_Knowledge))
        instance["input_ids_wo_know"] = list(chain(*sequence_wo_Knowledge))
        instance["lm_labels"] = [self.bos] + sequence_with_speaker[-1] + [self.eos]

        return instance, sequence

class ResponseGenerationDatasetEncoderDecoder_neg(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationDatasetEncoderDecoder_neg, self).__init__(args, tokenizer, split_type, labels, labels_file)
    
    def __getitem__(self, index):
        example = self.examples[index]
        this_inst = {
            "dialog_id": example["dialog_id"],
            "input_ids_w_know": []
        }

        candidates, label_idx = self.negative_sampling(example)
        for cand in candidates:
            instance, _ = self.build_input_from_segments(
                cand,
                example["history"],
                example["response"]
            )
            this_inst["input_ids_w_know"].append(instance["input_ids_w_know"])

        this_inst["input_ids_wo_know"] = instance["input_ids_wo_know"]
        this_inst["lm_labels"] = instance["lm_labels"]
        this_inst["response_text"] = example["response_text"]
        this_inst["knowledge_text"] = example["knowledge_text"]
        this_inst["label_idx"] = label_idx
        return this_inst
    
    def negative_sampling(self, example):
        candidates = list(self.snippets.keys())
        candidates = [self.snippets[cand_key] for cand_key in candidates]
        candidates = self._shrink_label_cands(example["knowledge"], candidates)
        label_idx = candidates.index(example["knowledge"])

        return candidates, label_idx

    def _shrink_label_cands(self, label, candidates):
        shrunk_label_cands = candidates.copy()
        shrunk_label_cands.remove(label)
        shrunk_label_cands = random.sample(shrunk_label_cands, k=4-1)
        shrunk_label_cands.append(label)
        random.shuffle(shrunk_label_cands)

        return shrunk_label_cands
    
    def collate_fn(self, batch):
        input_ids_w_know = [ids for ins in batch for ids in ins["input_ids_w_know"]]
        input_ids_wo_know = [ins["input_ids_wo_know"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]
        label_texts = [ins["response_text"] for ins in batch]
        knowledge_texts = [ins["knowledge_text"] for ins in batch]
        label_idx = [ins["label_idx"] for ins in batch]

        input_ids_w_know = torch.tensor(pad_ids(input_ids_w_know, self.pad))
        input_ids_wo_know = torch.tensor(pad_ids(input_ids_wo_know, self.pad))

        attention_mask_w_know = 1 - (input_ids_w_know == self.pad).int()
        attention_mask_wo_know = 1 - (input_ids_wo_know == self.pad).int()

        lm_labels = torch.tensor(pad_ids(lm_labels, -100))
        label_idx = torch.tensor(label_idx)

        return input_ids_w_know, input_ids_wo_know, \
            attention_mask_w_know, attention_mask_wo_know, \
            lm_labels, {'knowledge_texts':knowledge_texts, 'label_texts':label_texts, "contrastive_label_idx": label_idx}
    
    def build_input_from_segments(self, knowledge, history, response, with_eos=True):
        instance = {}
        
        sequence = [knowledge] + history + [response]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        history = list(chain(*sequence_with_speaker[:-1]))

        sequence_w_Knowledge = [[self.bos]] + [sequence[0]] + [[self.knowledge_tag]] + [history] + [[self.eos]]
        sequence_wo_Knowledge = [[self.bos]] + [[]] + [[self.knowledge_tag]] + [history] + [[self.eos]]

        instance["input_ids_w_know"] = list(chain(*sequence_w_Knowledge))
        instance["input_ids_wo_know"] = list(chain(*sequence_wo_Knowledge))
        instance["lm_labels"] = [self.bos] + sequence_with_speaker[-1] + [self.eos]

        return instance, sequence
class ResponseGenerationDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        rt_instance = dict()
        rt_instance["response_text"] = example["response_text"]
        rt_instance["knowledge_text"] = example["knowledge_text"]

        instance, _ = self.build_input_from_segments(
            example["knowledge"], 
            example["history"], 
            example["response"] if self.split_type == 'train' else [],
            with_eos=True if self.split_type == 'train' else False
        )
        rt_instance["input_ids_w_know"] = instance["input_ids"]
        rt_instance["lm_labels_w_know"] = instance["lm_labels"]

        instance, _ = self.build_input_from_segments(
            [], 
            example["history"], 
            example["response"] if self.split_type == 'train' else [],
            with_eos=True if self.split_type == 'train' else False
        )
        rt_instance["input_ids_wo_know"] = instance["input_ids"]
        rt_instance["lm_labels_wo_know"] = instance["lm_labels"]
        return rt_instance

    def collate_fn(self, batch):
        input_ids_w_know = [ins["input_ids_w_know"] for ins in batch]
        input_ids_wo_know = [ins["input_ids_wo_know"] for ins in batch]

        lm_labels_w_know = [ins["lm_labels_w_know"] for ins in batch]
        lm_labels_wo_know = [ins["lm_labels_wo_know"] for ins in batch]

        input_ids_w_know = torch.tensor(pad_ids(input_ids_w_know, self.pad))
        input_ids_wo_know = torch.tensor(pad_ids(input_ids_wo_know, self.pad))
        
        attention_mask_w_know = 1 - (input_ids_w_know == self.pad).int()
        attention_mask_wo_know = 1 - (input_ids_wo_know == self.pad).int()

        lm_labels_w_know = torch.tensor(pad_ids(lm_labels_w_know, -100))
        lm_labels_wo_know = torch.tensor(pad_ids(lm_labels_wo_know, -100))

        label_texts = [ins["response_text"] for ins in batch]
        knowledge_texts = [ins["knowledge_text"] for ins in batch]
        # we only want to use input_ids_wo_know and in order to match the return number of ResponseGenerationDatasetEncoderDecoder, we only return attention_mask_wo_know
        if self.args.use_external_in_evaluate:
            return input_ids_w_know, input_ids_wo_know, attention_mask_w_know,\
                lm_labels_w_know, lm_labels_wo_know,\
                {'knowledge_texts':knowledge_texts, 'label_texts':label_texts, "batchsize": len(batch)}
        else:
            return input_ids_w_know, input_ids_wo_know, attention_mask_wo_know,\
                lm_labels_w_know, lm_labels_wo_know,\
                {'knowledge_texts':knowledge_texts, 'label_texts':label_texts, "batchsize": len(batch)}

class ResponseGenerationDataset_Multilabel(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationDataset_Multilabel, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        rt_instance = {"input_ids_w_know":[],
                       "lm_labels_w_know":[],
                       "input_ids_wo_know":[],
                       "lm_labels_wo_know":[],
                       "response_text": example["response_text"],
                       "knowledge_text": example["knowledge_text"]
                       }
        response_list= example["response"] if self.split_type == 'train' else [[]]

        instance_list = [self.build_input_from_segments(
            example["knowledge"],
            example["history"],
            res,
            with_eos=True if self.split_type == 'train' else False)[0] for res in response_list
        ]

        for instance in instance_list:
            rt_instance["input_ids_w_know"].append(instance['input_ids'])
            rt_instance["lm_labels_w_know"].append(instance['lm_labels'])


        instance_list = [self.build_input_from_segments(
            [],
            example["history"],
            res,
            with_eos=True if self.split_type == 'train' else False)[0] for res in response_list
        ]
        for instance in instance_list:
            rt_instance["input_ids_wo_know"].append(instance["input_ids"])
            rt_instance["lm_labels_wo_know"].append(instance["lm_labels"])
        return rt_instance

    def collate_fn(self, batch):
        input_ids_w_know = [ids for ins in batch for ids in ins["input_ids_w_know"]]
        input_ids_wo_know = [ids for ins in batch for ids in ins["input_ids_wo_know"]]

        lm_labels_w_know = [ids for ins in batch for ids in ins["lm_labels_w_know"]]
        lm_labels_wo_know = [ids for ins in batch for ids in ins["lm_labels_wo_know"]]

        input_ids_w_know = torch.tensor(pad_ids(input_ids_w_know, self.pad))
        input_ids_wo_know = torch.tensor(pad_ids(input_ids_wo_know, self.pad))
        
        attention_mask_w_know = 1 - (input_ids_w_know == self.pad).int()
        attention_mask_wo_know = 1 - (input_ids_wo_know == self.pad).int()

        lm_labels_w_know = torch.tensor(pad_ids(lm_labels_w_know, -100))
        lm_labels_wo_know = torch.tensor(pad_ids(lm_labels_wo_know, -100))

        label_texts = [ids for ins in batch for ids in ins["response_text"]]
        knowledge_texts = [ins["knowledge_text"] for ins in batch]
        # we only want to use input_ids_wo_know and in order to match the return number of ResponseGenerationDatasetEncoderDecoder, we only return attention_mask_wo_know
        if self.args.use_external_in_evaluate:
            return input_ids_w_know, input_ids_wo_know, attention_mask_w_know,\
                lm_labels_w_know, lm_labels_wo_know,\
                {'knowledge_texts':knowledge_texts, 'label_texts':label_texts, "batchsize": len(batch)}
        else:
            return input_ids_w_know, input_ids_wo_know, attention_mask_wo_know,\
                lm_labels_w_know, lm_labels_wo_know,\
                {'knowledge_texts':knowledge_texts, 'label_texts':label_texts, "batchsize": len(batch)}

class ResponseGenerationEvalDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationEvalDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch


class KnowledgeSelectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeSelectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)
        if self.args.negative_sample_method not in ["all", "mix", "oracle"]:
            raise ValueError("negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)

    def _knowledge_to_string(self, doc, name=""):
        join_str = " %s " % self.knowledge_sep_token
        return join_str.join([name, doc["title"], doc["body"]])

    def __getitem__(self, index):
        example = self.examples[index]

        this_inst = {
            "dialog_id": example["dialog_id"],
            "input_ids": [],
            "token_type_ids": [],
            "mc_token_ids": []
        }

        if self.split_type != "train":
            # if eval_all_snippets is set, we use all snippets as candidates
            if self.args.eval_all_snippets:
                candidates = list(self.snippets.keys())
            else:
                candidates = example["candidates"]
        else:
            if self.args.negative_sample_method == "all":
                candidates = list(self.snippets.keys())
            elif self.args.negative_sample_method == "mix":
                candidates = example["candidates"] + random.sample(list(self.snippets.keys()), k=len(example["candidates"]))
            elif self.args.negative_sample_method == "oracle":
                candidates = example["candidates"]
            else: # although we have already checked for this, still adding this here to be sure
                raise ValueError("negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)
        
        candidate_keys = candidates
        this_inst["candidate_keys"] = candidate_keys
        candidates = [self.snippets[cand_key] for cand_key in candidates]

        if self.split_type == "train":
            candidates = self._shrink_label_cands(example["knowledge"], candidates)

        label_idx = candidates.index(example["knowledge"])
            
        this_inst["label_idx"] = label_idx
        for cand in candidates:
            instance, _ = self.build_input_from_segments(
                cand,
                example["history"]
            )
            this_inst["input_ids"].append(instance["input_ids"])
            this_inst["token_type_ids"].append(instance["token_type_ids"])
            this_inst["mc_token_ids"].append(instance["mc_token_ids"])

        return this_inst

    def build_input_from_segments(self, knowledge, history):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        instance = {}

        sequence = [[self.bos]] + history
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker + [[self.knowledge_tag] + knowledge + [self.eos]]

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence[:-1]) for _ in s] + [self.knowledge_tag for _ in sequence[-1]]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        return instance, sequence
    
    def _shrink_label_cands(self, label, candidates):
        shrunk_label_cands = candidates.copy()
        shrunk_label_cands.remove(label)
        shrunk_label_cands = random.sample(shrunk_label_cands, k=self.args.n_candidates-1)
        shrunk_label_cands.append(label)
        random.shuffle(shrunk_label_cands)

        return shrunk_label_cands

    def collate_fn(self, batch):
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
        mc_token_ids = [id for ins in batch for id in ins["mc_token_ids"]]
        label_idx = [ins["label_idx"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
            "candidate_keys": [ins["candidate_keys"] for ins in batch]
        }

        batch_size = len(batch)
        n_candidates = len(batch[0]["input_ids"])
        input_ids = torch.tensor(
            pad_ids(input_ids, self.pad)
        ).view(batch_size, n_candidates, -1)

        attention_mask = 1 - (input_ids == self.pad).int()
        
        token_type_ids = torch.tensor(
            pad_ids(token_type_ids, self.pad)
        ).view(batch_size, n_candidates, -1)

        lm_labels = torch.full_like(input_ids, -100)
        mc_token_ids = torch.tensor(mc_token_ids).view(batch_size, n_candidates)
        label_idx = torch.tensor(label_idx)

        return input_ids, token_type_ids, mc_token_ids, attention_mask, label_idx, data_info, {"batchsize": len(batch)}


class KnowledgeTurnDetectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeTurnDetectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def build_input_from_segments(self, history):
        """ Build a sequence of input from history """
        instance = {}

        sequence = [[self.bos]] + history[:-1] + [[self.knowledge_tag] + history[-1] + [self.eos]]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        return instance, sequence

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(example["history"])
        instance["label"] = example["knowledge_seeking"]
        instance["dialog_id"] = example["dialog_id"]
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        mc_token_ids = [ins["mc_token_ids"] for ins in batch]
        labels = [ins["label"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch]
        }

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        mc_token_ids = torch.tensor(mc_token_ids)
        lm_labels = torch.full_like(input_ids, -100)
        labels = torch.tensor(labels).float()

        return input_ids, token_type_ids, mc_token_ids, lm_labels, labels, data_info, {"batchsize": len(batch)}
