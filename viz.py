
import torch
import matplotlib.pyplot as plt

def plot_attention(cache, layer=None, head=0, token_index=None):
    attentions = cache["attentions"]
    tokens = cache["tokens"]

    num_layers = len(attentions)
    if layer is None:
        layer = num_layers // 2

    attn = attentions[layer][0]  # (H, T, T)

    if token_index is None:
        token_index = len(tokens) - 1

    scores = attn[head, token_index, :].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(max(6, len(tokens)/1.5), 3))
    im = ax.imshow(scores[None, :], aspect="auto")

    ax.set_yticks([])
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)

    ax.set_title(f"Attention — layer {layer}, head {head}, token {token_index}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_token_trajectory(cache, token_index=None, ref="first"):
    hidden_states = cache["hidden_states"]
    tokens = cache["tokens"]

    if token_index is None:
        token_index = len(tokens) - 1

    reps = torch.stack([h[0, token_index, :] for h in hidden_states], dim=0)
    reps = reps / (reps.norm(dim=-1, keepdim=True) + 1e-8)

    if ref == "first":
        ref_vec = reps[0]
        label = "layer 0"
    else:
        ref_vec = reps[-1]
        label = "final"

    cos = (reps * ref_vec).sum(dim=-1).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(hidden_states)), cos, marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Cosine sim to {label}")
    ax.set_title(f"Token trajectory for '{tokens[token_index]}'")
    ax.grid(True)
    plt.tight_layout()
    plt.show()
