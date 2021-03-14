LABEL_TO_TYPE = {
    "O": "Nontype",
    "X": "Cross",
    "I-ORG": "Organization",
    "B-ORG": "Organization",
    "B-DAT": "Date",
    "I-DAT": "Date",
    "B-ART": "Artifact",
    "I-ART": "Artifact",
    "B-MNY": "Money",
    "I-MNY": "Money",
    "B-TIM": "Time",
    "I-TIM": "Time",
    "B-PNT": "Percent",
    "I-PNT": "Percent",
    "B-PSN": "Person",
    "I-PSN": "Person",
    "B-LOC": "Location",
    "I-LOC": "Location",
}


def visualize(input_ids, result_ids, tokenizer, label_map):
    assert len(input_ids) == len(result_ids) == 128
    word_list = tokenizer.convert_ids_to_tokens(input_ids)
    # convert result_ids to label_list:
    label_list = []

    for result in result_ids:
        if result == 0:
            break
        label = label_map[result]
        label_list.append(label)

    output_cell = []
    n = len(label_list)

    # label to type

    cell_type = None
    cell_content = []

    for i in range(n):
        if word_list[i] == "[SEP]":
            break
        elif word_list[i] == '[CLS]':
            continue
        else:
            word = word_list[i]
            if word[0] == "#":
                # is a subword
                word = word[2:]

            label = label_list[i]
            part = LABEL_TO_TYPE[label]

            if label[0] == "B":
                if cell_type is not None:
                    cell = {
                        "type": cell_type,
                        "content": cell_content
                    }
                    output_cell.append(cell)
                    cell_content = []

                cell_type = LABEL_TO_TYPE[label]
                cell_content.append(word)

            elif label[0] == "I" or label == "X":
                # couldnt be a end of a ner
                cell_content.append(word)

            elif label == "O":
                if cell_type == LABEL_TO_TYPE[label]:
                    cell_content.append(word)
                else:
                    if cell_type is not None:
                        cell = {
                            "type": cell_type,
                            "content": cell_content
                        }
                        output_cell.append(cell)
                        cell_content = []

                    cell_type = LABEL_TO_TYPE[label]
                    cell_content.append(word)
            else:
                raise ValueError

    if cell_type is not None:
        cell = {
            "type": cell_type,
            "content": cell_content
        }
        output_cell.append(cell)

    for cell in output_cell:
        cell["content"] = "".join(cell["content"])

    return output_cell
