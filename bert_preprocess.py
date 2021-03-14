
def preprocess(text: str, label_map, max_seq_length, tokenizer):
    instances = []

    # split paragraph into sentences
    sentences = [r + "。" for r in text.strip().split("。")]

    # deal with one sentence
    for sent in sentences:
        sent = sent.split(' ')
        # label = text.label.split(' ')
        tokens = []
        labels = []
        for i, word in enumerate(sent):
            # print("word= ", word)
            token = tokenizer.tokenize(word)
            if '▁' in token:
                token.remove('▁')
            # print("token= ", token)
            tokens.extend(token)
            for m in range(len(token)):
                # dummy
                labels.append("O")

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        
        # append("O") or append("[CLS]") not sure!
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
            
        ntokens.append("[SEP]")
        segment_ids.append(0)
        # append("O") or append("[SEP]") not sure!
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        
        # label_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids.append(0)
            ntokens.append("[PAD]")

        # print(len(input_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        instance = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "label_ids": label_ids,
        }

        instances.append(instance)

    return instances
