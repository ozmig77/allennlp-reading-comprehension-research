import json
from reading_comprehension.data.drop_reader import DROPReader


def convert(drop_data_path,
            squad_style_output_path,
            skip_invalid,
            use_matched_span_as_answer_text):
    print(f"Converting: {drop_data_path}")
    reader = DROPReader(instance_format='squad',
                        skip_when_all_empty=["passage_span"] if skip_invalid else None,
                        relaxed_span_match_for_finding_labels=use_matched_span_as_answer_text)
    instances = reader.read(drop_data_path)
    print(f"Totally {len(list(instances))} instances")

    instances_grouped_by_passage = {}
    for instance in instances:
        passage_id = instance.fields["metadata"].metadata["passage_id"]
        if passage_id in instances_grouped_by_passage:
            instances_grouped_by_passage[passage_id].append(instance)
        else:
            instances_grouped_by_passage[passage_id] = [instance]

    squad_style_data = []
    for passage_id, instances in instances_grouped_by_passage.items():
        paragraph_text = instances[0].fields["metadata"].metadata["original_passage"]
        qas = []
        for instance in instances:
            metadata = instance.fields["metadata"].metadata
            gold_answer_text = metadata["answer_texts"][0]
            answer_spans = metadata["valid_passage_spans"]
            token_offsets = metadata["token_offsets"]
            answers = []
            for span in answer_spans:
                answer_start = token_offsets[span[0]][0]
                answer_end = token_offsets[span[1]][1]
                if use_matched_span_as_answer_text:
                    answer_text = paragraph_text[answer_start: answer_end]
                else:
                    answer_text = gold_answer_text
                answers.append({"answer_start": answer_start,
                                "text": answer_text})
                # print(paragraph_text[answer_start: answer_start + len(answer_text)])
                # print(answer_text)
            qas.append({"id": metadata["question_id"],
                        "question": metadata["original_question"],
                        "answers": [answers[0]]})
        new_passage = {"title": passage_id,
                       "paragraphs": [{"context": paragraph_text,
                                       "qas": qas}]}
        squad_style_data.append(new_passage)
    with open(squad_style_output_path, "w") as fout:
        squad_style_data = {"data": squad_style_data, "version": "drop-1.0"}
        json.dump(squad_style_data, fout)


def main():
    convert("drop_train.json",
            "drop_squad_style_train.json",
            skip_invalid=True,
            use_matched_span_as_answer_text=True)
    convert("drop_dev.json",
            "drop_squad_style_dev.json",
            skip_invalid=False,
            use_matched_span_as_answer_text=False)
    convert("drop_test.json",
            "drop_squad_style_test.json",
            skip_invalid=False,
            use_matched_span_as_answer_text=False)


if __name__ == "__main__":
    main()
