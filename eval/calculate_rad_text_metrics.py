# This script computes text-based retrieval metrics to
# compare generated and ground truth radiology reports.

# 1. BERTScore
#    Zhang et al., "BERTScore: Evaluating text generation with BERT", ICML 2020
#    Ref: https://openreview.net/pdf?id=SkeHuCVFDr

# 2. CheXbert Score
#    Zhang et al., "Optimizing the Factual Correctness of a Summary: A Study of Summarizing Radiology Reports", ACL 2020
#    Ref: https://aclanthology.org/2020.acl-main.458/

# 3. RadGraph Score
#    Delbrouck et al., "Improving the Factual Correctness of Radiology Report Generation with Semantic Reward", EMNLP 2022
#    Ref: https://aclanthology.org/2022.findings-emnlp.319.pdf 

# Script based on the VilMedic library by JB. Delbrouck
# https://github.com/jbdel/vilmedic

# requires packages radgraph, f1chexbert, bert_score
# available form pypi.org, through 'pip install radgraph f1chexbert bert_score'

import argparse

import os

import radgraph
import f1chexbert
import bert_score

import torch
import torch.nn as nn

from pprint import pprint

class BertScore(nn.Module):
    def __init__(self):
        super(BertScore, self).__init__()
        with torch.no_grad():
            self.bert_scorer = bert_score.BERTScorer(model_type='distilbert-base-uncased',
                                          num_layers=5,
                                          batch_size=64,
                                          nthreads=4,
                                          all_layers=False,
                                          idf=False,
                                          device='cuda',
                                          lang='en',
                                          rescale_with_baseline=True,
                                          baseline_path=None)

    def forward(self, refs, hyps):
        p, r, f = self.bert_scorer.score(
            cands=hyps,
            refs=refs,
            verbose=False,
            batch_size=64,
        )
        return torch.mean(f).item(), f.tolist()

def retrieve_strings_from_file(f):
    """Retrieve strings from file, one per line."""
    assert os.path.exists(f), f"File doesnt exist: {f}"
    return [line.strip() for line in open(f).readlines()]


def compute_f1radgraph(refs, hyps):
    """Compute F1RadGraph score."""
    assert len(refs) == len(hyps), "Number of references and hypotheses must match."
    res = radgraph.F1RadGraph(reward_level="all")(refs=refs, hyps=hyps)[0]
    return dict(
        radgraph_simple=res[0],
        radgraph_partial=res[1],
        radgraph_complete=res[2]
    )

def compute_chexbert(refs, hyps):
    """Compute F1CheXbert score."""
    assert len(refs) == len(hyps), "Number of references and hypotheses must match."
    res = f1chexbert.F1CheXbert(
        refs_filename=None,
        hyps_filename=None
    )(hyps, refs)
    return dict(
        accuracy=res[0],
        accuracy_per_sample=res[1],
        chexbert_all=res[2],
        chexbert_5=res[3]
    )

def compute_bert_score(refs, hyps):
    """Compute BERTScore score."""
    assert len(refs) == len(hyps), "Number of references and hypotheses must match."
    res = BertScore()(refs, hyps)
    return dict(
        bert_score=res[0],
        bert_score_per_sample=res[1]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("refs_file", type=str, help="Path to file containing references")
    parser.add_argument("hyps_file", type=str, help="Path to file containing hypotheses")
    parser.add_argument("--output_mode", type=str, default="reduced", choices=["reduced", "full"], help="Metrics reported")
    parser.add_argument("--output", type=str, default="scores.json", help="Path to output file")
    args = parser.parse_args()

    refs = retrieve_strings_from_file(args.refs_file)
    hyps = retrieve_strings_from_file(args.hyps_file)

    scores = dict(
        f1radgraph=compute_f1radgraph(refs, hyps),
        chexbert=compute_chexbert(refs, hyps),
        bert_score=compute_bert_score(refs, hyps),
    )

    reduced_scores = dict(
        f1radgraph=scores["f1radgraph"]["radgraph_partial"],
        f1chexbert_micro_avg=scores["chexbert"]["chexbert_5"]["micro avg"]["f1-score"],
        bert_score=scores["bert_score"]["bert_score"]
    )

    print("Computed scores:")
    pprint(reduced_scores)

    if args.output_mode == "reduced":
        scores = reduced_scores

    import json
    with open(args.output, "w") as f:
        json.dump(scores, f, indent=4)

if __name__ == "__main__":
    main()
