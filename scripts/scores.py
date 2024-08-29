import argparse
import json,ipdb
import os
import sys
sys.path.append('/path/to/this/repo')
import requests
from rouge_score import rouge_scorer
import summ_eval
#from summ_eval.bleu_metric import BleuMetric
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.bert_score_metric import BertScoreMetric

from dataset_walker import DatasetWalker

from multiprocessing import Pool
from summ_eval.metric import Metric
from typing import Sequence, Optional
from baseline.utils.metrics import KnowledgeF1

from sacrebleu.metrics import BLEU, BLEUScore
def corpus_bleu(hypotheses: Sequence[str],
                references: Sequence[Sequence[str]],
                smooth_method='exp',
                smooth_value=None,
                force=False,
                lowercase=False,
                max_ngram_order=4,
                tokenize=BLEU.TOKENIZER_DEFAULT,
                use_effective_order=False) -> BLEUScore:
    
    metric = BLEU(
        lowercase=lowercase, force=force, tokenize=tokenize,
        smooth_method=smooth_method, smooth_value=smooth_value,
        effective_order=use_effective_order, max_ngram_order=max_ngram_order)

    return metric.corpus_score(hypotheses, references)

class BleuMetric(Metric):
    def __init__(self, sent_smooth_method='exp', sent_smooth_value=None, sent_use_effective_order=True, \
       smooth_method='exp', smooth_value=None, force=False, lowercase=False, \
       use_effective_order=False, n_workers=24):

        self.sent_smooth_method = sent_smooth_method
        self.sent_smooth_value = sent_smooth_value
        self.sent_use_effective_order = sent_use_effective_order
        self.smooth_method = smooth_method
        self.smooth_value = smooth_value
        self.force = force
        self.lowercase = lowercase
        self.use_effective_order = use_effective_order
        self.n_workers = n_workers

    def evaluate_batch(self, summaries, references, aggregate=True ,max_ngram_order=4):
        if aggregate:
            if isinstance(references[0], str):
                references = [references]
            score = corpus_bleu(summaries, references, smooth_method=self.smooth_method, \
               smooth_value=self.smooth_value, force=self.force, lowercase=self.lowercase, \
               use_effective_order=self.use_effective_order, max_ngram_order=max_ngram_order)
            score_dict = {"bleu": score.score}
        else:
            p = Pool(processes=self.n_workers)
            score_dict = p.starmap(self.evaluate_example, zip(summaries, references))
            p.close()
        return score_dict

class Metric_my:
    def __init__(self, data_path):
        self.data_path = data_path
        self.reset()
        self.set_meteor()
        if 'wow' not in data_path and 'FaithDial' not in data_path:
            self.load_knowledge_corpus(data_path) 

    def reset(self):
        self._detection_tp = 0.0
        self._detection_fp = 0.0
        self._detection_fn = 0.0
        
        self._selection_tp = 0.0
        self._selection_fp = 0.0
        self._selection_fn = 0.0
        self._selection_exact_matched = 0.0
        self._selection_total = 0.0
        self._selection_mrr5 = 0.0
        self._selection_r1 = 0.0
        self._selection_r5 = 0.0

        self._generation_rouge_l = 0.0
        self._generation_rouge_1 = 0.0
        self._generation_rouge_2 = 0.0

        self._generation_KF1 = 0.0

        self._ref_responses = []
        self._pred_responses = []
        self._ref_knowledges = []

    def _match(self, ref_knowledge, pred_knowledge):
        result = []
        for pred in pred_knowledge:
            matched = False
            if 'wow' in self.data_path or 'FaithDial' in self.data_path:
                if ref_knowledge == pred:
                    matched = True
            else:
                for ref in ref_knowledge:
                    if pred['domain'] == ref['domain'] and pred['entity_id'] == ref['entity_id'] and pred['doc_id'] == ref['doc_id']:
                        matched = True
            result.append(matched)
        return result

    def _reciprocal_rank(self, ref_knowledge, hyp_knowledge, k=5):
        relevance = self._match(ref_knowledge, hyp_knowledge)[:k]

        if True in relevance:
            idx = relevance.index(True)
            result = 1.0/(idx+1)
        else:
            result = 0.0

        return result

    def _recall_at_k(self, ref_knowledge, hyp_knowledge, k=5):
        relevance = self._match(ref_knowledge, hyp_knowledge)[:k]

        if True in relevance:
            result = 1.0
        else:
            result = 0.0

        return result

    def set_meteor(self):
        file_path = summ_eval.__file__
        dir = os.path.dirname(file_path)
        if not os.path.exists(os.path.join(dir, "data")):
            os.mkdir(os.path.join(dir, "data"))
        if not os.path.exists(os.path.join(dir, "data", "paraphrase-en.gz")):
            paraphrase_en_gz_url = "https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/meteor/data/paraphrase-en.gz?raw=true"
            r = requests.get(paraphrase_en_gz_url)
            with open(os.path.join(dir, "data", "paraphrase-en.gz"), "wb") as outputf:
                outputf.write(r.content)
    
    def load_knowledge_corpus(self, data_path):
        try:
            with open(os.path.join(data_path,'knowledge.json'), 'r') as f:
                output = json.load(f)
        except FileNotFoundError:
            sys.exit('Knowledge file does not exist at %s' % os.path.join(data_path,'knowledge.json'))
        self.knowledge_corpus = output

    def _match_knowledge_obj(self, obj1, obj2):
        matched = False
        if obj2['domain'] == obj1['domain'] and obj2['entity_id'] == obj1['entity_id'] and obj2['doc_id'] == obj1['doc_id']:
            matched = True
        return matched

    def _remove_duplicate_knowledge(self, objs):
        result = []
        for obj_i in objs:
            duplicated = False
            for obj_j in result:
                if self._match_knowledge_obj(obj_i, obj_j) is True:
                    duplicated = True
            if duplicated is False:
                result.append(obj_i)
        return result

    def _match_knowledge(self, ref_objs, pred_objs):
        num_matched = 0
        for ref in ref_objs:
            for pred in pred_objs:
                if 'wow' in self.data_path or 'FaithDial' in self.data_path:
                    if ref == pred:
                        num_matched += 1
                elif self._match_knowledge_obj(ref, pred):
                    num_matched += 1

        tp = num_matched
        fp = len(pred_objs) - num_matched
        fn = len(ref_objs) - num_matched

        if len(ref_objs) == len(pred_objs) and len(ref_objs) == tp:
            exact_matched = 1
        else:
            exact_matched = 0

        return (tp, fp, fn, exact_matched)
    
    def _rouge(self, ref_response, hyp_response):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        scores = scorer.score(ref_response, hyp_response)

        rouge1 = scores['rouge1'].fmeasure
        rouge2 = scores['rouge2'].fmeasure
        rougeL = scores['rougeL'].fmeasure

        return {'rouge1': rouge1, 'rouge2': rouge2, 'rougeL': rougeL}

    def _knowledge_F1(self, ref_knowledge_idx, hyp_response):
        scorer = KnowledgeF1()

        if 'wow' in self.data_path or 'FaithDial' in self.data_path:
            self._ref_knowledges.append(ref_knowledge_idx)
            return scorer.f1_score(hyp_response, [ref_knowledge_idx])

        domain = ref_knowledge_idx[0]['domain']
        entity_id = ref_knowledge_idx[0]['entity_id']
        doc_id = ref_knowledge_idx[0]['doc_id']

        ref_knowledge = [self.knowledge_corpus[domain][f'{entity_id}']['docs'][f'{doc_id}']['body']]
        self._ref_knowledges.append(self.knowledge_corpus[domain][f'{entity_id}']['docs'][f'{doc_id}']['body'])

        scores = scorer.f1_score(hyp_response, ref_knowledge)
        return scores
                    
    def update(self, ref_obj, hyp_obj):
        if ref_obj['target'] is True:
            if hyp_obj['target'] is True:
                self._ref_responses.append(ref_obj['response'])
                self._pred_responses.append(hyp_obj['response'])
                
                self._detection_tp += 1

                self._selection_mrr5 += self._reciprocal_rank(ref_obj['knowledge'], hyp_obj['knowledge'], 5)
                self._selection_r1 += self._recall_at_k(ref_obj['knowledge'], hyp_obj['knowledge'], 1)
                self._selection_r5 += self._recall_at_k(ref_obj['knowledge'], hyp_obj['knowledge'], 5)

                rouge_scores = self._rouge(ref_obj['response'], hyp_obj['response'])
                self._generation_rouge_l += rouge_scores['rougeL']
                self._generation_rouge_1 += rouge_scores['rouge1']
                self._generation_rouge_2 += rouge_scores['rouge2']

                self._generation_KF1 += self._knowledge_F1(ref_obj['knowledge'], hyp_obj['response'])
            else:
                self._detection_fn += 1
        else:
            if hyp_obj['target'] is True:
                self._detection_fp += 1

    def _compute(self, score_sum):
        if self._detection_tp + self._detection_fp > 0.0:
            score_p = score_sum/(self._detection_tp + self._detection_fp)
        else:
            score_p = 0.0

        if self._detection_tp + self._detection_fn > 0.0:
            score_r = score_sum/(self._detection_tp + self._detection_fn)
        else:
            score_r = 0.0

        if score_p + score_r > 0.0:
            score_f = 2*score_p*score_r/(score_p+score_r)
        else:
            score_f = 0.0

        return (score_p, score_r, score_f)
        
    def scores(self):
        detection_p, detection_r, detection_f = self._compute(self._detection_tp)

        if self._selection_tp + self._selection_fp > 0:
            selection_p = self._selection_tp / (self._selection_tp + self._selection_fp)
        else:
            selection_p = 0.0

        if self._selection_tp + self._selection_fn > 0:
            selection_r = self._selection_tp / (self._selection_tp + self._selection_fn)
        else:
            selection_r = 0.0

        if selection_p + selection_r > 0.0:
            selection_f = 2 * selection_p * selection_r / (selection_p + selection_r)
        else:
            selection_f = 0.0

        # selection_em_acc = self._selection_exact_matched / self._selection_total
        selection_mrr5_p, selection_mrr5_r, selection_mrr5_f = self._compute(self._selection_mrr5)
        selection_r1_p, selection_r1_r, selection_r1_f = self._compute(self._selection_r1)
        selection_r5_p, selection_r5_r, selection_r5_f = self._compute(self._selection_r5)

        bleu_metric = BleuMetric()
        
        bleu_score_1 = bleu_metric.evaluate_batch(self._pred_responses, self._ref_responses, max_ngram_order=1)['bleu'] / 100.0 * self._detection_tp
        bleu_score_2 = bleu_metric.evaluate_batch(self._pred_responses, self._ref_responses, max_ngram_order=2)['bleu'] / 100.0 * self._detection_tp
        bleu_score_3 = bleu_metric.evaluate_batch(self._pred_responses, self._ref_responses, max_ngram_order=3)['bleu'] / 100.0 * self._detection_tp
        bleu_score_4 = bleu_metric.evaluate_batch(self._pred_responses, self._ref_responses, max_ngram_order=4)['bleu'] / 100.0 * self._detection_tp

        meteor_metric = MeteorMetric()
        meteor_score = meteor_metric.evaluate_batch(self._pred_responses, self._ref_responses)['meteor'] * self._detection_tp

        bert_score_metric = BertScoreMetric(model_type='/root/roberta-large')
        bert_score = bert_score_metric.evaluate_batch(self._pred_responses, self._ref_knowledges)['bert_score_f1'] * self._detection_tp

        generation_bleu_1_p, generation_bleu_1_r, generation_bleu_1_f = self._compute(bleu_score_1)
        generation_bleu_2_p, generation_bleu_2_r, generation_bleu_2_f = self._compute(bleu_score_2)
        generation_bleu_3_p, generation_bleu_3_r, generation_bleu_3_f = self._compute(bleu_score_3)
        generation_bleu_4_p, generation_bleu_4_r, generation_bleu_4_f = self._compute(bleu_score_4)

        generation_meteor_p, generation_meteor_r, generation_meteor_f = self._compute(meteor_score)

        generation_rouge_l_p, generation_rouge_l_r, generation_rouge_l_f = self._compute(self._generation_rouge_l)
        generation_rouge_1_p, generation_rouge_1_r, generation_rouge_1_f = self._compute(self._generation_rouge_1)
        generation_rouge_2_p, generation_rouge_2_r, generation_rouge_2_f = self._compute(self._generation_rouge_2)

        generation_KF1_p, generation_KF1_r, generation_KF1_f = self._compute(self._generation_KF1)

        generation_bert_score_p, generation_bert_score_r, generation_bert_score_f = self._compute(bert_score)

        scores = {
            'detection': {
                'prec': detection_p,
                'rec': detection_r,
                'f1': detection_f
            },
            'selection': {
                'prec': selection_p,
                'rec': selection_r,
                'f1': selection_f,
                'mrr@5': selection_mrr5_f,
                'r@1': selection_r1_f,
                'r@5': selection_r5_f,
                # 'em_acc': selection_em_acc
            },
            'generation': {
                'bleu-1': generation_bleu_1_f,
                'bleu-2': generation_bleu_2_f,
                'bleu-3': generation_bleu_3_f,
                'bleu-4': generation_bleu_4_f,
                'meteor': generation_meteor_f,
                'rouge_1': generation_rouge_1_f,
                'rouge_2': generation_rouge_2_f,
                'rouge_l': generation_rouge_l_f,
                'KF1': generation_KF1_f,
                'bert_score': generation_bert_score_f
            }
        }

        return scores
        
def main(argv):
    parser = argparse.ArgumentParser(description='Evaluate the system outputs.')

    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', choices=['train', 'val', 'val_seen', 'val_unseen', 'test', 'test_seen', 'test_unseen'], required=True, help='The dataset to analyze')
    parser.add_argument('--dataroot',dest='dataroot',action='store', metavar='PATH', required=True,
                        help='Will look for corpus in <dataroot>/<dataset>/...')
    parser.add_argument('--outfile',dest='outfile',action='store',metavar='JSON_FILE',required=True,
                        help='File containing output JSON')
    parser.add_argument('--scorefile',dest='scorefile',action='store',metavar='JSON_FILE',required=True,
                        help='File containing scores')

    args = parser.parse_args()

    try:
        with open(args.outfile, 'r') as f:
            output = json.load(f)
    except FileNotFoundError:
        sys.exit('Output file does not exist at %s' % args.outfile)

    data = DatasetWalker(dataroot=args.dataroot, dataset=args.dataset, labels=True)

    metric = Metric_my(args.dataroot)

    if len(data) != len(output):
        raise ValueError("the number of instances between ground truth and output does not match")

    for (instance, ref), pred in zip(data, output):
        metric.update(ref, pred)
        
    scores = metric.scores()

    with open(args.scorefile, 'w') as out:
        json.dump(scores, out, indent=2)
    

if __name__ =="__main__":
    main(sys.argv)        
