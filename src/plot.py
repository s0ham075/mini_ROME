import matplotlib.pyplot as plt
import torch

def plot_trace_heatmap(matrix, token_labels, next_word, min_prob, subj_range, color=None, title=None):
    """Plots heatmap of restoration effects across layers/tokens."""
    if color is None:
        color = "Purples"
    token_labels = token_labels.copy()
    start, end = subj_range
    for i in range(start - 1, end - 1):
        token_labels[i] = token_labels[i] + '*'

    plt.figure(figsize=(12, 5))
    plt.imshow(matrix, aspect='auto', cmap=color, vmin=0, vmax=1)
    plt.colorbar(label=f'p({next_word})')
    plt.yticks(range(len(token_labels)), token_labels)
    plt.xticks(range(matrix.shape[1]), list(range(matrix.shape[1])))
    if title:
        plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Token")
    plt.tight_layout()
    plt.show()


def plot_aie_heatmap(matrix, color=None, title=None):
    """Plots heatmap for Average Indirect Effects across layers/tokens."""
    if color is None:
        color = "Purples"
    y_labels = ['First Subject', 'Last Subject', 'Next Token', 'Last Token']

    plt.figure(figsize=(12, 5))
    plt.imshow(matrix, aspect='auto', cmap=color, vmin=0, vmax=0.2)
    plt.colorbar(label='AIE')
    plt.yticks(range(len(y_labels)), y_labels)
    plt.xticks(range(matrix.shape[1]), list(range(matrix.shape[1])))
    if title:
        plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Token")
    plt.tight_layout()
    plt.show()


def plot_aie_effect(model, facts, total_aie_effects_fn):
    """Compute total AIE effects and plot heatmaps."""
    result = total_aie_effects_fn(model, facts)
    scores = result['resid_table'].detach().cpu()

    print("AIE Resid Table:", scores)
    plot_aie_heatmap(scores, color="Purples", title="AIE Residual States")
