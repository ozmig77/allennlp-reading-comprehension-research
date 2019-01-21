import string
import re
from typing import Tuple
from overrides import overrides
from allennlp.tools import squad_eval
from allennlp.training.metrics.metric import Metric


STOPWORDS = set(["a", "an", "the"])
PUNCTUATIONS = set(string.punctuation)


def string_to_bag(raw_text):
    text = raw_text.lower()
    text_tokens = set()
    for token in text.strip().split(" "):
        if not re.match(r"\d*\.\d+", token):
            token = ''.join(ch for ch in token if ch not in PUNCTUATIONS)
        if token != '':
            text_tokens.add(token)
    return set(text_tokens) - STOPWORDS


def bag_of_words_exact_match(prediction: str, ground_truth: str):
    return string_to_bag(prediction) == string_to_bag(ground_truth)


def bag_of_words_f1(prediction: str, ground_truth: str):
    prediction_bag = string_to_bag(prediction)
    gold_bag = string_to_bag(ground_truth)
    hit = len(gold_bag.intersection(prediction_bag))
    if hit > 0:
        precision = 1.0 * hit / len(prediction_bag)
        recall = 1.0 * hit / len(gold_bag)
        return 2.0 * precision * recall / (precision + recall)
    else:
        return 0.0


@Metric.register("drop")
class DropEmAndF1(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computed exact match and F1 score based on bag of words
    """
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    @overrides
    def __call__(self, best_span_string, answer_strings):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        exact_match = squad_eval.metric_max_over_ground_truths(
                bag_of_words_exact_match,
                best_span_string,
                answer_strings)
        f1_score = squad_eval.metric_max_over_ground_truths(
                bag_of_words_f1,
                best_span_string,
                answer_strings)
        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official SQuAD script
        over all inputs.
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return exact_match, f1_score

    @overrides
    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __str__(self):
        return f"DropEmAndF1(em={self._total_em}, f1={self._total_f1})"
