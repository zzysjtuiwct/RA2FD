# Code for RA2FD
1. run External_knowledge_train.sh to train a teacher model.
2. run KnowledgeInjection_Mistral.py to inject knowledge into language model.
3. run Pure_Decoder_train.sh to finetune Mistral.
4. run Parameter_Knowledge_Multilabel_train.sh to finetune Mistral using RA2FD
5. Replace Mistral with Llama or replace wow with DSTC9 dataset to run different experiments.