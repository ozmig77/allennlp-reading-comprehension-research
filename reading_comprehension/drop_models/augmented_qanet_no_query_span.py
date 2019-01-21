from typing import Any, Dict, List, Optional

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.bidaf import BidirectionalAttentionFlow
from allennlp.modules import Highway
from allennlp.nn.activations import Activation
from allennlp.modules.feedforward import FeedForward
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from reading_comprehension.utils import memory_effient_masked_softmax as masked_softmax
from reading_comprehension.drop_em_and_f1 import DropEmAndF1


@Model.register("augmented_qanet_no_q")
class AugmentedQANetNoQuestion(Model):
    """
    This class adapts the QANet model to do question answering on DROP dataset.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 matrix_attention_layer: MatrixAttention,
                 modeling_layer: Seq2SeqEncoder,
                 dropout_prob: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        text_embed_dim = text_field_embedder.get_output_dim()
        encoding_in_dim = phrase_layer.get_input_dim()
        encoding_out_dim = phrase_layer.get_output_dim()
        modeling_in_dim = modeling_layer.get_input_dim()
        modeling_out_dim = modeling_layer.get_output_dim()

        self._text_field_embedder = text_field_embedder

        self._embedding_proj_layer = torch.nn.Linear(text_embed_dim, encoding_in_dim)
        self._highway_layer = Highway(encoding_in_dim, num_highway_layers)

        self._encoding_proj_layer = torch.nn.Linear(encoding_in_dim, encoding_in_dim)
        self._phrase_layer = phrase_layer

        self._matrix_attention = matrix_attention_layer

        self._modeling_proj_layer = torch.nn.Linear(encoding_out_dim * 4, modeling_in_dim)
        self._modeling_layer = modeling_layer

        self._passage_weights_predictor = torch.nn.Linear(modeling_out_dim, 1)
        self._question_weights_predictor = torch.nn.Linear(encoding_out_dim, 1)

        self._answer_type_predictor = FeedForward(modeling_out_dim + encoding_out_dim,
                                                  activations=[Activation.by_name('relu')(),
                                                               Activation.by_name('linear')()],
                                                  hidden_dims=[modeling_out_dim, 3],
                                                  num_layers=2,
                                                  dropout=dropout_prob)

        self._passage_span_start_predictor = FeedForward(modeling_out_dim * 2,
                                                         activations=[Activation.by_name('relu')(),
                                                                      Activation.by_name('linear')()],
                                                         hidden_dims=[modeling_out_dim, 1],
                                                         num_layers=2)
        self._passage_span_end_predictor = FeedForward(modeling_out_dim * 2,
                                                       activations=[Activation.by_name('relu')(),
                                                                    Activation.by_name('linear')()],
                                                       hidden_dims=[modeling_out_dim, 1],
                                                       num_layers=2)

        self._number_sign_predictor = FeedForward(modeling_out_dim * 3,
                                                  activations=[Activation.by_name('relu')(),
                                                               Activation.by_name('linear')()],
                                                  hidden_dims=[modeling_out_dim, 3],
                                                  num_layers=2)
        self._count_number_predictor = FeedForward(modeling_out_dim,
                                                   activations=[Activation.by_name('relu')(),
                                                                Activation.by_name('linear')()],
                                                   hidden_dims=[modeling_out_dim, 10],
                                                   num_layers=2)
        self._drop_metrics = DropEmAndF1()
        self._dropout = torch.nn.Dropout(p=dropout_prob)

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                numbers_in_passage: Dict[str, torch.LongTensor],
                number_indices: torch.LongTensor,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                answer_as_plus_minus_combinations: torch.LongTensor = None,
                answer_as_counts: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ, unused-argument

        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        embedded_question = self._dropout(self._text_field_embedder(question))
        embedded_passage = self._dropout(self._text_field_embedder(passage))
        embedded_question = self._highway_layer(self._embedding_proj_layer(embedded_question))
        embedded_passage = self._highway_layer(self._embedding_proj_layer(embedded_passage))

        batch_size = embedded_question.size(0)

        projected_embedded_question = self._encoding_proj_layer(embedded_question)
        projected_embedded_passage = self._encoding_proj_layer(embedded_passage)

        encoded_question = self._dropout(self._phrase_layer(projected_embedded_question, question_mask))
        encoded_passage = self._dropout(self._phrase_layer(projected_embedded_passage, passage_mask))

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = masked_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # Shape: (batch_size, question_length, passage_length)
        question_passage_attention = masked_softmax(passage_question_similarity.transpose(1, 2), passage_mask)
        # Shape: (batch_size, question_length, encoding_dim)
        # question_passage_vectors = util.weighted_sum(encoded_passage, question_passage_attention)

        # Shape: (batch_size, passage_length, passage_length)
        passsage_attention_over_attention = torch.bmm(passage_question_attention, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_passage_vectors = util.weighted_sum(encoded_passage, passsage_attention_over_attention)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        merged_passage_attention_vectors = self._dropout(
                torch.cat([encoded_passage, passage_question_vectors,
                           encoded_passage * passage_question_vectors,
                           encoded_passage * passage_passage_vectors],
                          dim=-1))

        modeled_passage_list = [self._modeling_proj_layer(merged_passage_attention_vectors)]
        for _ in range(4):
            modeled_passage = self._dropout(self._modeling_layer(modeled_passage_list[-1], passage_mask))
            modeled_passage_list.append(modeled_passage)

        passage_weights = self._passage_weights_predictor(modeled_passage_list[1]).squeeze(-1)
        passage_weights = masked_softmax(passage_weights, passage_mask)
        passage_vector = util.weighted_sum(modeled_passage_list[1], passage_weights)
        question_weights = self._question_weights_predictor(encoded_question).squeeze(-1)
        question_weights = masked_softmax(question_weights, question_mask)
        question_vector = util.weighted_sum(encoded_question, question_weights)

        # Shape: (batch_size, 4)
        answer_type_logits = self._answer_type_predictor(torch.cat([passage_vector, question_vector], -1))
        answer_type_log_probs = torch.nn.functional.log_softmax(answer_type_logits, -1)

        # Shape: (batch_size, 10)
        count_number_logits = self._count_number_predictor(passage_vector)
        count_number_log_probs = torch.nn.functional.log_softmax(count_number_logits, -1)

        # Shape: (batch_size, passage_length, modeling_dim * 2))
        passage_for_span_start = torch.cat([modeled_passage_list[1], modeled_passage_list[2]], dim=-1)
        # Shape: (batch_size, passage_length)
        passage_span_start_logits = self._passage_span_start_predictor(passage_for_span_start).squeeze(-1)
        # Shape: (batch_size, passage_length, modeling_dim * 2)
        passage_for_span_end = torch.cat([modeled_passage_list[1], modeled_passage_list[3]], dim=-1)
        # Shape: (batch_size, passage_length)
        passage_span_end_logits = self._passage_span_end_predictor(passage_for_span_end).squeeze(-1)
        # Shape: (batch_size, passage_length)
        passage_span_start_log_probs = util.masked_log_softmax(passage_span_start_logits, passage_mask)
        passage_span_end_log_probs = util.masked_log_softmax(passage_span_end_logits, passage_mask)

        # Shape: (batch_size, # of numbers in the passage)
        number_indices = number_indices.squeeze(-1)
        number_mask = (number_indices != -1).float()
        clamped_number_indices = torch.nn.functional.relu(number_indices)
        encoded_passage_for_numbers = torch.cat([modeled_passage_list[1], modeled_passage_list[4]], dim=-1)
        # Shape: (batch_size, # of numbers in the passage, encoding_dim)
        encoded_numbers = \
            torch.gather(encoded_passage_for_numbers,
                         1,
                         clamped_number_indices.unsqueeze(-1).expand(-1, -1, encoded_passage_for_numbers.size(-1)))
        encoded_numbers = encoded_numbers * number_mask.unsqueeze(-1)
        # Shape: (batch_size, # of numbers in the passage)
        encoded_numbers = \
            torch.cat([encoded_numbers, passage_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)], -1)

        # Shape: (batch_size, # of numbers in the passage, 3)
        number_sign_logits = self._number_sign_predictor(encoded_numbers)
        number_sign_log_probs = torch.nn.functional.log_softmax(number_sign_logits, -1)

        passage_span_start_logits = util.replace_masked_values(passage_span_start_logits, passage_mask, -1e7)
        passage_span_end_logits = util.replace_masked_values(passage_span_end_logits, passage_mask, -1e7)
        # Shape: (batch_size, 2)
        best_passage_span = \
            BidirectionalAttentionFlow.get_best_span(passage_span_start_logits, passage_span_end_logits)
        # Shape: (batch_size, 2)
        best_passage_start_log_probs = \
            torch.gather(passage_span_start_log_probs, 1, best_passage_span[:, 0].unsqueeze(-1)).squeeze(-1)
        best_passage_end_log_probs = \
            torch.gather(passage_span_end_log_probs, 1, best_passage_span[:, 0].unsqueeze(-1)).squeeze(-1)
        # Shape: (batch_size,)
        best_passage_span_log_prob = best_passage_start_log_probs + best_passage_end_log_probs
        best_passage_span_log_prob += answer_type_log_probs[:, 0]

        # Shape: (batch_size, # of numbers in passage).
        # For padding numbers, the best sign masked as 0 (not included).
        best_signs_for_numbers = torch.argmax(number_sign_log_probs, -1) * number_mask.long()
        # Shape: (batch_size, # of numbers in passage)
        best_signs_log_probs = \
            torch.gather(number_sign_log_probs, 2, best_signs_for_numbers.unsqueeze(-1)).squeeze(-1)
        # the probs of the masked positions should be 1 so that it will not affect the joint probability
        # TODO: this is not quite right, since if there are many numbers in the passage,
        # TODO: the joint probability would be very small.
        best_signs_log_probs = util.replace_masked_values(best_signs_log_probs, number_mask, 0)
        # Shape: (batch_size,)
        best_combination_log_prob = best_signs_log_probs.sum(-1)
        best_combination_log_prob += answer_type_log_probs[:, 1]

        # Shape: (batch_size,)
        best_count_number = torch.argmax(count_number_log_probs, -1)
        best_count_log_prob = torch.gather(count_number_log_probs, 1, best_count_number.unsqueeze(-1)).squeeze(-1)
        best_count_log_prob += answer_type_log_probs[:, 2]

        # best_answer_type = \
        #     torch.argmax(torch.stack([best_passage_span_log_prob,
        #                               best_question_span_log_prob,
        #                               best_combination_log_prob,
        #                               best_count_log_prob],
        #                              1),
        #                  1)
        #
        best_answer_type = torch.argmax(answer_type_log_probs, 1)

        output_dict = {}

        # If answer is given, compute the loss for training.
        if answer_as_passage_spans is not None:
            # Shape: (batch_size, # of answer spans)
            passage_span_starts = answer_as_passage_spans[:, :, 0]
            passage_span_ends = answer_as_passage_spans[:, :, 1]
            # Some spans are padded with index -1, so we need to mask them
            passage_span_mask = (passage_span_starts != -1).float()
            clamped_passage_span_starts = torch.nn.functional.relu(passage_span_starts)
            clamped_passage_span_ends = torch.nn.functional.relu(passage_span_ends)
            # Shape: (batch_size, # of answer spans)
            log_likelihood_for_passage_span_starts = \
                torch.gather(passage_span_start_log_probs, 1, clamped_passage_span_starts)
            log_likelihood_for_passage_span_ends = \
                torch.gather(passage_span_end_log_probs, 1, clamped_passage_span_ends)
            # Shape: (batch_size, # of answer spans)
            log_likelihood_for_passage_spans = \
                log_likelihood_for_passage_span_starts + log_likelihood_for_passage_span_ends
            # For those padded spans, we set their log probabilities to be very small negative value
            log_likelihood_for_passage_spans = \
                util.replace_masked_values(log_likelihood_for_passage_spans, passage_span_mask, -1e10)
            # Shape: (batch_size, )
            log_marginal_likelihood_for_passage_spans = \
                answer_type_log_probs[:, 0] + util.logsumexp(log_likelihood_for_passage_spans)

            # Some combinations are padded with the labels of all numbers as 0, so we mask them here
            # Shape: (batch_size, # of combinations)
            combination_mask = (answer_as_plus_minus_combinations.sum(-1) > 0).float()
            # Shape: (batch_size, # of numbers in the passage, # of combinations)
            combinations = answer_as_plus_minus_combinations.transpose(1, 2)
            # Shape: (batch_size, # of numbers in the passage, # of combinations)
            log_likelihood_for_number_signs = torch.gather(number_sign_log_probs, 2, combinations)
            # the log likelihood of the masked positions should be 0
            # so that it will not affect the joint probability
            log_likelihood_for_number_signs = \
                util.replace_masked_values(log_likelihood_for_number_signs, number_mask.unsqueeze(-1), 0)
            # Shape: (batch_size, # of combinations)
            log_likelihood_for_combinations = log_likelihood_for_number_signs.sum(1)
            # For those padded combinations, we set their log probabilities to be very small negative value
            log_likelihood_for_combinations = \
                util.replace_masked_values(log_likelihood_for_combinations, combination_mask, -1e10)
            # Shape: (batch_size, )
            log_marginal_likelihood_for_combination = \
                answer_type_log_probs[:, 1] + util.logsumexp(log_likelihood_for_combinations)

            # Some count answers are padded with label -1, we should mask them
            # Shape: (batch_size, # of count answers)
            count_mask = (answer_as_counts != -1).float()
            # Shape: (batch_size, # of count answers)
            clamped_counts = torch.nn.functional.relu(answer_as_counts)
            log_likelihood_for_counts = torch.gather(count_number_log_probs, 1, clamped_counts)
            # For those padded spans, we set their log probabilities to be very small negative value
            log_likelihood_for_counts = \
                util.replace_masked_values(log_likelihood_for_counts, count_mask, -1e10)
            # Shape: (batch_size, )
            log_marginal_likelihood_for_count = \
                answer_type_log_probs[:, 2] + util.logsumexp(log_likelihood_for_counts)

            marginal_log_likelihood = util.logsumexp(
                    torch.stack([log_marginal_likelihood_for_passage_spans,
                                 log_marginal_likelihood_for_combination,
                                 log_marginal_likelihood_for_count],
                                dim=-1))

            output_dict["loss"] = - marginal_log_likelihood.mean()

        # Compute the metrics and add the tokenized input to the output.
        if metadata is not None:
            output_dict["question_id"] = []
            output_dict["answer"] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                answer_type = best_answer_type[i].detach().cpu().numpy()

                # We did not consider multi-mention answers here
                if answer_type == 0:  # passage span answer
                    passage_str = metadata[i]['original_passage']
                    offsets = metadata[i]['passage_token_offsets']
                    predicted_span = tuple(best_passage_span[i].detach().cpu().numpy())
                    start_offset = offsets[predicted_span[0]][0]
                    end_offset = offsets[predicted_span[1]][1]
                    best_answer_str = passage_str[start_offset:end_offset]
                elif answer_type == 1:  # plus_minus combination answer
                    original_numbers = metadata[i]['original_numbers']
                    sign_remap = {0: 0, 1: 1, 2: -1}
                    predicted_signs = [sign_remap[it] for it in best_signs_for_numbers[i].detach().cpu().numpy()]
                    result = sum([sign * number for sign, number in zip(predicted_signs, original_numbers)])
                    best_answer_str = str(result)
                elif answer_type == 2:
                    predicted_count = best_count_number[i].detach().cpu().numpy()
                    best_answer_str = str(predicted_count)
                else:
                    raise ValueError(f"Answer type should be 0, 1, 2 or 3, but got {answer_type}")

                output_dict["question_id"].append(metadata[i]["question_id"])
                output_dict["answer"].append(best_answer_str)
                answer_texts = metadata[i].get('answer_texts', [])
                if answer_texts:
                    self._drop_metrics(best_answer_str, answer_texts)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        return {'em': exact_match, 'f1': f1_score}
