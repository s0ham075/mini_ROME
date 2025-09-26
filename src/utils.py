import torch
from functools import partial

def predict_next_word(model, logits):
    """Return top predicted token and its probability."""
    top_token_id = logits.argmax().item()
    top_token = model.to_string(top_token_id)
    return top_token, logits[top_token_id].item()


def find_correct_probability(model, logits, attribute):
    """Find probability of the correct next token."""
    answer_token = model.to_single_token(attribute)
    return logits[answer_token]


def get_std_embeddings(model, known_facts):
    """Compute std dev of subject embeddings."""
    subject_vectors = []
    for fact in known_facts:
        tokens = model.to_tokens(fact['subject'], prepend_bos=True)
        cache = model.run_with_cache(tokens, remove_batch_dim=True, stop_at_layer=0)
        subject_vectors.append(cache[0][0])

    subject_vectors = torch.cat(subject_vectors, dim=0)
    return subject_vectors.std().item()


def find_subject_token_range(model, token_array, substring):
    """Find start/end token indices of a substring in a tokenized prompt."""
    toks = model.to_str_tokens(token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def add_noise_hook(value, hook, subj_pos=None, noise_level=1.0):
    """Adds Gaussian noise to activations at given positions."""
    s_pos, e_pos = subj_pos
    noise = torch.randn(value.shape[0] - 1, e_pos - s_pos, value.shape[2]) * noise_level
    noise = noise.to(value.device)
    value[1:, s_pos:e_pos] += noise
    return value


def replace_correct_activaton(value, hook, token_no):
    """Restores activations for a given token to clean state."""
    value[1:, token_no] = value[0, token_no]
    return value
