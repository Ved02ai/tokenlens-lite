
import torch
import torch.nn.functional as F

@torch.no_grad()
def _run_full_and_cache(model, tokenizer, prompt, device):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    out_prefix = model(input_ids[:, :-1], use_cache=True)
    past_kv = out_prefix.past_key_values

    out_full = model(input_ids[:, -1:], past_key_values=past_kv, use_cache=True)
    logits_full = out_full.logits[:, -1, :]

    return {"input_ids": input_ids, "past_kv": past_kv, "logits_full": logits_full}


def _prune_past_kv(past_kv, keep_last):
    pruned = []
    for k, v in past_kv:
        pruned.append((k[:, :, -keep_last:, :], v[:, :, -keep_last:, :]))
    return tuple(pruned)


@torch.no_grad()
def compare_with_kv_pruning(model, tokenizer, prompt, keep_fraction=0.4, device=None, top_k=5):
    if device is None:
        device = next(model.parameters()).device

    cache = _run_full_and_cache(model, tokenizer, prompt, device)
    input_ids = cache["input_ids"]
    past_kv = cache["past_kv"]
    logits_full = cache["logits_full"]

    T = input_ids.size(1)
    keep_last = max(1, int((T - 1) * keep_fraction))

    pruned = _prune_past_kv(past_kv, keep_last)

    out_pruned = model(input_ids[:, -1:], past_key_values=pruned, use_cache=True)
    logits_pruned = out_pruned.logits[:, -1, :]

    p_full = F.softmax(logits_full.float(), dim=-1)
    p_pruned = F.softmax(logits_pruned.float(), dim=-1)

    l1_dist = torch.sum(torch.abs(p_full - p_pruned)).item()

    top1_full_prob, top1_id = torch.max(p_full, dim=-1)
    prob_shift = abs(top1_full_prob[0].item() - p_pruned[0, top1_id[0]].item())

    top_full = torch.topk(p_full, top_k, dim=-1)
    top_pruned = torch.topk(p_pruned, top_k, dim=-1)

    return {
        "keep_fraction": keep_fraction,
        "l1_dist": l1_dist,
        "top1_token": tokenizer.decode(top1_id[0].item()),
        "top1_prob_shift": prob_shift,
        "top_full_tokens": [(tokenizer.decode(i), float(v)) for i, v in zip(top_full.indices[0], top_full.values[0])],
        "top_pruned_tokens": [(tokenizer.decode(i), float(v)) for i, v in zip(top_pruned.indices[0], top_pruned.values[0])],
    }
