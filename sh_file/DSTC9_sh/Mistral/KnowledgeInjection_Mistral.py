import torch,ipdb,random
import numpy as np
import argparse, os
from argparse import Namespace
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    BartForConditionalGeneration,
    GPT2LMHeadModel,
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq,
    AutoModelForCausalLM,
    EvalPrediction
)

from torch.utils.data import DataLoader, RandomSampler
from scripts.knowledge_reader import KnowledgeReader
from baseline.dataset import (
    KnowledgeInjectionDataset,
    KnowledgeInjectionDataset_puredecoder,
    SPECIAL_TOKENS
)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    ipdb.set_trace()
    return {"accuracy": accuracy_score(p.label_ids, preds)}

def main():
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--warmup_ratio", type=float, default=0.2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # set seed
    set_seed(args)

    # load model
    model_name_or_path = "/path/to/Mistral-7B-v0.1"
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config, device_map="auto", torch_dtype =  torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))

    # load dataset
    dataroot = 'data_eval'
    knowledge_file = 'knowledge.json'
    train_dataset = KnowledgeInjectionDataset_puredecoder(tokenizer, dataroot, knowledge_file)
    RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, collate_fn=train_dataset.collate_fn, shuffle=True, batch_size=args.train_batch_size
    )

    # build trainer
    training_args = TrainingArguments(
        save_strategy="epoch",
        output_dir="runs/inject_model/Mistral"+str(args.num_train_epochs),
        evaluation_strategy="no",
        logging_steps=1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        push_to_hub=False,
        save_total_limit=1,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model, padding = 'longest', return_tensors = 'pt')
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

if __name__ == "__main__":
    main()