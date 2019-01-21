import torch
from allennlp.data.tokenizers import Token


def memory_effient_masked_softmax(vector: torch.Tensor, mask: torch.Tensor,
                                  dim: int = -1, mask_value=-1e7) -> torch.Tensor:
    """
    This is an approximate version of `allennlp.nn.util.masked_softmax`.
    By using less operations here than the original `masked_softmax`, we save a lot of memory.
    But you should be careful that this function does not return an array of ``0.0``, as the
    original `mask_softmax` does, in the case that the input vector is completely masked.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # To limit numerical errors from large vector elements outside the mask, we zero these out.
        result = torch.nn.functional.softmax(vector + (1 - mask) * mask_value, dim=dim)
    return result


def split_tokens_by_hyphen(tokens):
    hyphens = ["-", "–", "~"]
    new_tokens = []

    def split_token_by_hyphen(token, hyphen):
        split_tokens = []
        char_offset = token.idx
        for sub_str in token.text.split(hyphen):
            if sub_str:
                split_tokens.append(Token(text=sub_str, idx=char_offset))
                char_offset += len(sub_str)
            split_tokens.append(Token(text=hyphen, idx=char_offset))
            char_offset += len(hyphen)
        if split_tokens:
            split_tokens.pop(-1)
            char_offset -= len(hyphen)
            return split_tokens
        else:
            return [token]

    for token in tokens:
        if any(hyphen in token.text for hyphen in hyphens):
            unsplit_tokens, split_tokens = [token], []
            for hyphen in hyphens:
                for unsplit_token in unsplit_tokens:
                    if hyphen in token.text:
                        split_tokens += split_token_by_hyphen(unsplit_token, hyphen)
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)

    return new_tokens
