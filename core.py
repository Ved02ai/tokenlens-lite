
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name: str = "gpt2", device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    return model, tokenizer, device


@torch.no_grad()
def run_with_cache(model, tokenizer, prompt: str, device: str | None = None):
    if device is None:
        device = next(model.parameters()).device

    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    outputs = model(
        **enc,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
    )

    attentions = outputs.attentions
    if attentions is None or attentions[0] is None:
        raise ValueError("Model did not return attention tensors.")

    input_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    cache = {
        "prompt": prompt,
        "input_ids": input_ids,
        "tokens": tokens,
        "logits": outputs.logits,
        "hidden_states": outputs.hidden_states,
        "attentions": attentions,
        "model_name": getattr(model.config, "_name_or_path", "unknown"),
    }
    return cache
