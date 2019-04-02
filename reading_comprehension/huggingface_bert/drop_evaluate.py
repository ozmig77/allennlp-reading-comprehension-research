# !/usr/bin/python

import argparse
import json
import re
import string

import numpy as np


# Copied from: https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        if not is_number(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        else:
            return text

    def lower(text):
        return text.lower()

    def tokenize(text):
        return re.split(" |-", text)

    sp = [white_space_fix(remove_articles(str(normalize_number(remove_punc(lower(tok)))))) for tok in tokenize(s)]
    sp = [s for s in sp if s.strip()]
    sp = ' '.join(sp).strip()
    return sp


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def normalize_number(text):
    if is_number(text):
        return float(text)
    else:
        return text


def answer_to_bags(answer):
    span_bag = set()
    raw_spans = []
    if isinstance(answer, list) or isinstance(answer, tuple):
        raw_spans = answer
    if isinstance(answer, str):
        raw_spans = [answer]
    span_bag = set()
    token_bag = []
    for raw_span in raw_spans:
        span = normalize_answer(raw_span)
        span_bag.add(span)
        token_bag.append(set(span.split()))
    return span_bag, token_bag


def align_bags(predicted, gold):
    f1_scores = []
    for gold_it in range(len(gold)):
        gold_item = gold[gold_it]
        max_f1 = 0.0
        max_it = None
        best_alignment = (set([]), set([]))
        for pred_it in range(len(predicted)):
            pred_item = predicted[pred_it]
            current_f1 = compute_f1(pred_item, gold_item)
            if current_f1 >= max_f1:
                best_alignment = (gold_item, pred_item)
                max_f1 = current_f1
                max_it = pred_it
        match_flag = match_numbers_if_present(*best_alignment)
        if match_flag:
            f1_scores.append(max_f1)
        else:
            f1_scores.append(0.0)
        gold[gold_it] = {}
        predicted[max_it] = {}
    return f1_scores


def compute_f1(predicted_bag, gold_bag):
    intersection = len(gold_bag.intersection(predicted_bag))
    if len(predicted_bag) == 0:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if len(gold_bag) == 0:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return f1


def get_metrics(predicted, gold):
    predicted_bags = answer_to_bags(predicted)
    gold_bags = answer_to_bags(gold)

    exact_match = 1.0 if predicted_bags[0] == gold_bags[0] else 0

    f1_per_bag = align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def match_numbers_if_present(gold_bag, predicted_bag):
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if is_number(word):
            predicted_numbers.add(word)
    if len(gold_numbers) == 0 or (len(gold_numbers) > 0 and len(gold_numbers.intersection(predicted_numbers)) > 0):
        return True
    return False


def to_string(answer):
    if answer["number"] != "":
        return tuple([str(answer["number"])]), "number"
    elif len(answer["spans"]) > 0:
        return tuple(answer["spans"]), "span" if len(answer["spans"]) == 1 else "spans"
    else:
        return tuple(
            ["{0} {1} {2}".format(answer["date"]["day"], answer["date"]["month"], answer["date"]["year"])]), "date"


def _run_evaluation(annotations, predicted_answers):
    """
    Evaluation for programatic use.
    """
    exact_match = []
    f1 = []
    # for each type as well
    type_to_em = {}
    type_to_f1 = {}
    for pid, annotation in annotations.items():
        for qa_pair in annotation["qa_pairs"]:
            query_id = qa_pair["query_id"]
            max_em_score = 0
            max_f1_score = 0
            max_type = None
            if query_id in predicted_answers:
                predicted = predicted_answers[query_id]
            else:
                print("Missing prediction for question: {}".format(query_id))
                predicted = None
            for answer in [qa_pair["answer"]] + qa_pair["validated_answers"]:
                gold_answer, gold_type = to_string(answer)
                em_score, f1_score = get_metrics(predicted, gold_answer)
                if gold_answer[0].strip() != "":
                    max_em_score = max(max_em_score, em_score)
                    max_f1_score = max(max_f1_score, f1_score)
                    if max_em_score == em_score or max_f1_score == f1_score: max_type = gold_type
            exact_match.append(max_em_score)
            f1.append(max_f1_score)
            if max_type not in type_to_em:
                type_to_em[max_type] = []
            type_to_em[max_type].append(max_em_score)
            if max_type not in type_to_f1:
                type_to_f1[max_type] = []
            type_to_f1[max_type].append(max_f1_score)

    global_em = np.mean(exact_match)
    global_f1 = np.mean(f1)
    print("Exact-match accuracy {0:.2f}".format(global_em * 100))
    print("F1 score {0:.2f}".format(global_f1 * 100))
    print("{0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))
    print("----")
    total = np.sum([len(v) for v in type_to_em.values()])
    for typ in sorted(type_to_em.keys()):
        print("{0}: {1} ({2:.2f}%)".format(typ, len(type_to_em[typ]), 100. * len(type_to_em[typ]) / total))
        print("  Exact-match accuracy {0:.3f}".format(100. * np.mean(type_to_em[typ])))
        print("  F1 score {0:.3f}".format(100. * np.mean(type_to_f1[typ])))
    # return global_em, global_f1
    return {"em": global_em, "f1": global_f1}


def run_evaluation(prediction_path, gold_path):
    predicted_answers = json.load(open(prediction_path, encoding='utf-8'))
    annotations = json.load(open(gold_path, encoding='utf-8'))
    return _run_evaluation(annotations, predicted_answers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluate on drop dataset')
    parser.add_argument("--gold_path", type=str, required=False, default="drop_dataset_test.gold.json",
                        help='location of the gold file')
    parser.add_argument("--prediction_path", type=str, required=False, default="sample_predictions.json",
                        help='location of the prediction file')
    args = parser.parse_args()
    run_evaluation(args.prediction_path, args.gold_path)
