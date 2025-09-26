import torch
from functools import partial
from .utils import (
    add_noise_hook,
    replace_correct_activaton,
    find_correct_probability,
    predict_next_word,
    find_subject_token_range
)
import transformer_lens.utils as utils


def trace_important_window_tflens(model, input_tokens, subj_index, correct_next_word,
                                  window=10, kind="resid"):
    """Trace importance across tokens/layers using windowed activation restoration."""
    num_layers = model.cfg.n_layers
    temp_noise_hook_fn = partial(add_noise_hook, subj_pos=subj_index)

    table = []
    for tok in range(len(input_tokens[0])):
        if tok == 0:
            continue
        row = []
        for center_layer in range(num_layers):
            window_layers = range(max(0, center_layer - window // 2),
                                  min(num_layers, center_layer + window // 2 + 1))

            fwd_hooks = [(utils.get_act_name('embed'), temp_noise_hook_fn)]
            for L in window_layers:
                hook_name = {
                    "mlp": utils.get_act_name("mlp_out", L),
                    "attn": utils.get_act_name("attn_out", L),
                    "resid": utils.get_act_name("resid_pre", L)
                }.get(kind, None)
                if hook_name is None:
                    raise ValueError(f"Unknown kind {kind}")
                fwd_hooks.append((hook_name, partial(replace_correct_activaton, token_no=tok)))

            run = model.run_with_hooks(input_tokens, return_type='logits', fwd_hooks=fwd_hooks)
            logits = run[:, -1]
            logits = torch.softmax(logits, dim=1).mean(dim=0)
            row.append(find_correct_probability(model, logits, correct_next_word))
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_states(model, fact):
    """Run clean/corrupted experiments and trace important tokens/layers."""
    prompts = [fact['prompt'] for _ in range(11)]
    input_tokens = model.to_tokens(prompts, prepend_bos=True)
    subj_index = find_subject_token_range(model, input_tokens[0], fact['subject'])

    # Clean run
    clean_logits = torch.softmax(model(input_tokens, return_type='logits')[:, -1], dim=1).mean(dim=0)
    correct_next_word, max_prob = predict_next_word(model, clean_logits)

    # Corrupted run
    temp_hook_fn = partial(add_noise_hook, subj_pos=subj_index)
    corrupted_logits = model.run_with_hooks(input_tokens, return_type='logits',
                                            fwd_hooks=[(utils.get_act_name('embed'), temp_hook_fn)])[1:, -1]
    corrupted_logits = torch.softmax(corrupted_logits, dim=1).mean(dim=0)
    min_prob = find_correct_probability(model, corrupted_logits, correct_next_word)

    # Restore activations at each token/layer
    table = []
    for tok in range(len(input_tokens[0])):
        if tok == 0: continue
        row = []
        for layer in range(model.cfg.n_layers):
            run = model.run_with_hooks(input_tokens, return_type='logits',
                                       fwd_hooks=[(utils.get_act_name('embed'), temp_hook_fn),
                                                  (utils.get_act_name("resid_pre", layer),
                                                   partial(replace_correct_activaton, token_no=tok))])
            logits = torch.softmax(run[1:, -1], dim=1).mean(dim=0)
            row.append(find_correct_probability(model, logits, correct_next_word))
        table.append(torch.stack(row))

    resid_table = torch.stack(table)
    mlp_table = trace_important_window_tflens(model, input_tokens, subj_index, correct_next_word, kind='mlp')
    attn_table = trace_important_window_tflens(model, input_tokens, subj_index, correct_next_word, kind='attn')

    return {
        'resid_table': resid_table,
        'mlp_table': mlp_table,
        'attn_table': attn_table,
        'max_prob': max_prob,
        'min_prob': min_prob,
        'correct_next_word': correct_next_word,
    }


def calculate_aie_effect(model, fact):
    """Compute Average Indirect Effect for given fact across layers/tokens."""
    prompts = [fact['prompt'] for _ in range(2)]
    input_tokens = model.to_tokens(prompts, prepend_bos=True)
    subj_index = find_subject_token_range(model, input_tokens[0], fact['subject'])

    clean_logits = torch.softmax(model(input_tokens, return_type='logits')[:, -1], dim=1).mean(dim=0)
    correct_next_word, max_prob = predict_next_word(model, clean_logits)

    temp_hook_fn = partial(add_noise_hook, subj_pos=subj_index)
    corrupted_logits = model.run_with_hooks(input_tokens, return_type='logits',
                                            fwd_hooks=[(utils.get_act_name('embed'), temp_hook_fn)])[1:, -1]
    corrupted_logits = torch.softmax(corrupted_logits, dim=1).mean(dim=0)
    min_prob = find_correct_probability(model, corrupted_logits, correct_next_word)

    key_tokens = [subj_index[0], subj_index[1] - 1, subj_index[1], len(input_tokens[0]) - 1]

    table = []
    for tok in key_tokens:
        row = []
        for layer in range(model.cfg.n_layers):
            run = model.run_with_hooks(input_tokens, return_type='logits',
                                       fwd_hooks=[(utils.get_act_name('embed'), temp_hook_fn),
                                                  (utils.get_act_name("resid_pre", layer),
                                                   partial(replace_correct_activaton, token_no=tok))])
            logits = torch.softmax(run[1:, -1], dim=1).mean(dim=0)
            row.append(find_correct_probability(model, logits, correct_next_word) - min_prob)
        table.append(torch.stack(row))

    return {
        'resid_table': torch.stack(table),
        'max_prob': max_prob,
        'min_prob': min_prob,
        'correct_next_word': correct_next_word
    }


def total_aie_effects(model, facts):
    resid_total, mlp_total, attn_total = None, None, None

    for fact in facts:
        effect = calculate_aie_effect(model, fact)
        resid_total = effect['resid_table'] if resid_total is None else resid_total + effect['resid_table']

    if resid_total is not None:
        resid_total /= len(facts)

    return {'resid_table': resid_total}
