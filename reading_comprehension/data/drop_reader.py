import json
import logging
import itertools
from typing import Dict, List, Union, Tuple, Any
from collections import defaultdict
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension.util import make_reading_comprehension_instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.dataset_readers.reading_comprehension.util import IGNORED_TOKENS, STRIPPED_CHARACTERS
from allennlp.data.fields import Field, TextField, MetadataField, LabelField, ListField, \
    SequenceLabelField, SpanField, IndexField
from reading_comprehension.utils import split_tokens_by_hyphen

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# TODO: Add more number here
WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}


@DatasetReader.register("drop")
class DROPReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 relaxed_span_match: bool = True,
                 do_augmentation: bool = True,
                 passage_length_limit: int = None,
                 question_length_limit: int = None,
                 passage_length_limit_for_evaluation: int = None,
                 question_length_limit_for_evaluation: int = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._relaxed_span_match = relaxed_span_match
        self._do_augmentation = do_augmentation
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.passage_length_limit_for_eval = passage_length_limit_for_evaluation or passage_length_limit
        self.question_length_limit_for_eval = question_length_limit_for_evaluation or question_length_limit

    @overrides
    def _read(self, file_path: str):
        # pylint: disable=logging-fstring-interpolation
        # if `file_path` is a URL, redirect to the cache
        is_train = "train" in str(file_path)
        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        logger.info("Reading the dataset")
        instances, skip_count = [], 0
        max_passage_len = self.passage_length_limit if is_train else self.passage_length_limit_for_eval
        max_question_len = self.question_length_limit if is_train else self.question_length_limit_for_eval
        for passage_id, passage_info in dataset.items():
            passage_text = passage_info["passage"]
            passage_tokens = self._tokenizer.tokenize(passage_text)
            passage_tokens = split_tokens_by_hyphen(passage_tokens)
            for question_answer in passage_info["qa_pairs"]:
                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                answer_annotation = question_answer["answer"] if "answer" in question_answer else None
                instance = self.text_to_instance(question_text,
                                                 passage_text,
                                                 question_id,
                                                 passage_id,
                                                 answer_annotation,
                                                 passage_tokens,
                                                 max_passage_len,
                                                 max_question_len,
                                                 drop_invalid=is_train)
                if instance is not None:
                    instances.append(instance)
                else:
                    skip_count += 1
        logger.info(f"Skipped {skip_count} questions, kept {len(instances)} questions.")
        return instances

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         question_id: str = None,
                         passage_id: str = None,
                         answer_annotation: Dict[str, Union[str, Dict, List]] = None,
                         passage_tokens: List[Token] = None,
                         max_passage_len: int = None,
                         max_question_len: int = None,
                         drop_invalid: bool = False) -> Union[Instance, None]:
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
            passage_tokens = split_tokens_by_hyphen(passage_tokens)
        question_tokens = self._tokenizer.tokenize(question_text)
        question_tokens = split_tokens_by_hyphen(question_tokens)
        # passage_text = question_text
        # passage_tokens = question_tokens
        if max_passage_len is not None:
            passage_tokens = passage_tokens[: max_passage_len]
        if max_question_len is not None:
            question_tokens = question_tokens[: max_question_len]

        answer_type, answer_texts = None, []
        if answer_annotation is not None:
            answer_type, answer_texts = self.convert_answer(answer_annotation)

        # The original answer_texts is a list, because we treat the all spans and all date tokens as correct.
        # This is useful when finding the matched spans for training.
        # However, for evaluation, all tokens should be regard as one single answer.
        answer_texts_for_evaluation = [' '.join(answer_texts)]
        if not self._relaxed_span_match:
            answer_texts = answer_texts_for_evaluation

        # Tokenize the answer text in order to find the matched span based on token
        tokenized_answer_texts = []
        for answer_text in answer_texts:
            answer_tokens = self._tokenizer.tokenize(answer_text)
            answer_tokens = split_tokens_by_hyphen(answer_tokens)
            tokenized_answer_texts.append(' '.join(token.text for token in answer_tokens))

        if not self._do_augmentation:
            valid_passage_spans = \
                self.find_valid_spans(passage_tokens, tokenized_answer_texts) if tokenized_answer_texts else []
            if not valid_passage_spans and drop_invalid:
                return None
            if not valid_passage_spans:
                valid_passage_spans = [(0, 0)]
            return make_reading_comprehension_instance(question_tokens,
                                                       passage_tokens,
                                                       self._token_indexers,
                                                       passage_text,
                                                       valid_passage_spans,
                                                       answer_texts_for_evaluation,
                                                       additional_metadata={
                                                               "original_passage": passage_text,
                                                               "original_question": question_text,
                                                               "passage_id": passage_id,
                                                               "question_id": question_id,
                                                               "valid_passage_spans": valid_passage_spans})
        else:
            numbers_in_passage = []
            number_indices = []
            for token_index, token in enumerate(passage_tokens):
                number = self.convert_string_to_int(token.text)
                if number is not None:
                    numbers_in_passage.append(number)
                    number_indices.append(token_index)
            # hack to guarantee minimal length of padded number
            numbers_in_passage.append(0)
            number_indices.append(-1)
            numbers_as_tokens = [Token(str(number)) for number in numbers_in_passage]
            valid_passage_spans = \
                self.find_valid_spans(passage_tokens, tokenized_answer_texts) if tokenized_answer_texts else []
            valid_question_spans = \
                self.find_valid_spans(question_tokens, tokenized_answer_texts) if tokenized_answer_texts else []
            target_numbers = []
            for answer_text in answer_texts:
                number = self.convert_string_to_int(answer_text)
                if number is not None:
                    target_numbers.append(number)
            valid_plus_minus_combinations = []
            if answer_type in ["number", "date"]:
                valid_plus_minus_combinations = \
                    self.find_valid_plus_minus_combinations(numbers_in_passage, target_numbers)
            valid_counts = []
            if answer_type == "number":
                numbers_for_count = list(range(10))
                valid_counts = self.find_valid_count(numbers_for_count, target_numbers)

            if not valid_passage_spans \
                    and not valid_question_spans \
                    and not valid_plus_minus_combinations \
                    and not valid_counts \
                    and drop_invalid:
                return None

            answer_info = {"answer_texts": answer_texts_for_evaluation,
                           "answer_passage_spans": valid_passage_spans,
                           "answer_question_spans": valid_question_spans,
                           "plus_minus_combinations": valid_plus_minus_combinations,
                           "counts": valid_counts}

            return self.make_augmented_instance(question_tokens,
                                                passage_tokens,
                                                numbers_as_tokens,
                                                number_indices,
                                                self._token_indexers,
                                                passage_text,
                                                answer_info,
                                                additional_metadata={
                                                        "original_passage": passage_text,
                                                        "original_question": question_text,
                                                        "original_numbers": numbers_in_passage,
                                                        "passage_id": passage_id,
                                                        "question_id": question_id,
                                                        "answer_info": answer_info})

    @staticmethod
    def make_augmented_instance(question_tokens: List[Token],
                                passage_tokens: List[Token],
                                number_tokens: List[Token],
                                number_indices: List[int],
                                token_indexers: Dict[str, TokenIndexer],
                                passage_text: str,
                                answer_info: Dict[str, Any] = None,
                                additional_metadata: Dict[str, Any] = None) -> Instance:
        additional_metadata = additional_metadata or {}
        fields: Dict[str, Field] = {}
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        question_offsets = [(token.idx, token.idx + len(token.text)) for token in question_tokens]

        # This is separate so we can reference it later with a known type.
        fields["passage"] = TextField(passage_tokens, token_indexers)
        fields["question"] = TextField(question_tokens, token_indexers)
        number_index_fields = [IndexField(index, fields["passage"]) for index in number_indices]
        fields["number_indices"] = ListField(number_index_fields)
        # This field is actually not required in the model,
        # it is used to create the `answer_as_plus_minus_combinations` field, which is a `SequenceLabelField`.
        # We cannot use `number_indices` field for creating that, because the `ListField` will not be empty
        # when we want to create a new empty field. That will lead to error.
        fields["numbers_in_passage"] = TextField(number_tokens, token_indexers)
        metadata = {"original_passage": passage_text,
                    "passage_token_offsets": passage_offsets,
                    "question_token_offsets": question_offsets,
                    "question_tokens": [token.text for token in question_tokens],
                    "passage_tokens": [token.text for token in passage_tokens],
                    "number_tokens": [token.text for token in number_tokens],
                    "number_indices": number_indices}
        if answer_info:
            metadata["answer_texts"] = answer_info["answer_texts"]

            passage_span_fields = \
                [SpanField(span[0], span[1], fields["passage"]) for span in answer_info["answer_passage_spans"]]
            if not passage_span_fields:
                passage_span_fields.append(SpanField(-1, -1, fields["passage"]))
            fields["answer_as_passage_spans"] = ListField(passage_span_fields)

            question_span_fields = \
                [SpanField(span[0], span[1], fields["question"]) for span in answer_info["answer_question_spans"]]
            if not question_span_fields:
                question_span_fields.append(SpanField(-1, -1, fields["question"]))
            fields["answer_as_question_spans"] = ListField(question_span_fields)

            plus_minus_combinations_fields = []
            for plus_minus_combination in answer_info["plus_minus_combinations"]:
                plus_minus_combinations_fields.append(
                        SequenceLabelField(plus_minus_combination, fields["numbers_in_passage"]))
            if not plus_minus_combinations_fields:
                plus_minus_combinations_fields.append(
                        SequenceLabelField([0] * len(fields["numbers_in_passage"]), fields["numbers_in_passage"]))
            fields["answer_as_plus_minus_combinations"] = ListField(plus_minus_combinations_fields)

            count_fields = [LabelField(count_label, skip_indexing=True) for count_label in answer_info["counts"]]
            if not count_fields:
                count_fields.append(LabelField(-1, skip_indexing=True))
            fields["answer_as_counts"] = ListField(count_fields)

        metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    @staticmethod
    def convert_string_to_int(string: str):
        no_comma_string = string.replace(",", "")
        if no_comma_string in WORD_NUMBER_MAP:
            number = WORD_NUMBER_MAP[no_comma_string]
        else:
            try:
                number = int(no_comma_string)
            except ValueError:
                number = None
        return number

    @staticmethod
    def find_valid_spans(passage_tokens: List[Token],
                         answer_texts: List[str]) -> List[Tuple[int, int]]:
        normalized_tokens = [token.text.lower().strip(STRIPPED_CHARACTERS) for token in passage_tokens]
        word_positions: Dict[str, List[int]] = defaultdict(list)
        for i, token in enumerate(normalized_tokens):
            word_positions[token].append(i)
        spans = []
        for answer_text in answer_texts:
            answer_tokens = answer_text.lower().strip(STRIPPED_CHARACTERS).split()
            num_answer_tokens = len(answer_tokens)
            if answer_tokens[0] not in word_positions:
                continue
            for span_start in word_positions[answer_tokens[0]]:
                span_end = span_start  # span_end is _inclusive_
                answer_index = 1
                while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                    token = normalized_tokens[span_end + 1]
                    if answer_tokens[answer_index].strip(STRIPPED_CHARACTERS) == token:
                        answer_index += 1
                        span_end += 1
                    elif token in IGNORED_TOKENS:
                        span_end += 1
                    else:
                        break
                if num_answer_tokens == answer_index:
                    spans.append((span_start, span_end))
        return spans

    @staticmethod
    def find_valid_plus_minus_combinations(numbers: List[int],
                                           targets: List[int],
                                           max_length_of_combinations: int = 2) -> List[List[int]]:
        valid_combinations = []
        for combination_length in range(2, max_length_of_combinations + 1):
            possible_signs = list(itertools.product((-1, 1), repeat=combination_length))
            for combination in itertools.combinations(enumerate(numbers), combination_length):
                indices = [it[0] for it in combination]
                values = [it[1] for it in combination]
                for signs in possible_signs:
                    eval_value = sum(sign * value for sign, value in zip(signs, values))
                    if eval_value in targets:
                        labels_for_numbers = [0] * len(numbers)  # 0 represents ``not included''.
                        for index, sign in zip(indices, signs):
                            labels_for_numbers[index] = 1 if sign == 1 else 2  # 1 for positive, 2 for negative
                        valid_combinations.append(labels_for_numbers)
        return valid_combinations

    @staticmethod
    def find_valid_count(count_numbers: List[int],
                         targets: List[int]) -> List[int]:
        valid_indices = []
        for index, number in enumerate(count_numbers):
            if number in targets:
                valid_indices.append(index)
        return valid_indices

    @staticmethod
    def convert_answer(answer_annotation: Dict[str, Union[str, Dict, List]]) -> Tuple[str, List]:
        answer_type = None
        if answer_annotation["spans"]:
            answer_type = "spans"
        elif answer_annotation["number"]:
            answer_type = "number"
        elif any(answer_annotation["date"].values()):
            answer_type = "date"

        answer_content = answer_annotation[answer_type] if answer_type is not None else None

        answer_texts = []
        if answer_type is None:  # No answer
            pass
        elif answer_type == "spans":
            # answer_content is a list of string in this case
            answer_texts = answer_content
        elif answer_type == "date":
            # answer_content is a dict with "month", "day", "year" as the keys
            date_tokens = [answer_content[key]
                           for key in ["month", "day", "year"] if key in answer_content and answer_content[key]]
            answer_texts = date_tokens
        elif answer_type == "number":
            # answer_content is a string of number
            answer_texts = [answer_content]
        return answer_type, answer_texts
