class InputFeatures(object):
    """
    Inputs of model.
    """
    def __init__(self, tokens, input_ids, input_mask, input_type_ids):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, tokenizer):
    """
    Convert textual segments into word IDs.

    params
        examples: the raw textual segments in a list.
        tokenizer: a BERT Tokenizer object.

    return
        features: BERT features in a list.
    """
    features = []  
    for (ex_index, example) in enumerate(examples):
        new_tokens = []
        input_type_ids = []

        new_tokens.append("[CLS]")
        input_type_ids.append(0)
        for sentence in example:
            for word in sentence:
                new_tokens.append(word)
                input_type_ids.append([0])
        new_tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(new_tokens)
        input_mask = [1] * len(input_ids)

        print(new_tokens)

        features.append(
            InputFeatures(
                tokens=new_tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features
