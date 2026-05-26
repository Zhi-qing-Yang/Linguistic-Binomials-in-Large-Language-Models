import torch


@torch.no_grad()
def continuation_logprob(model, tokenizer, prefix, continuation, device):
    if not continuation:
        return 0.0

    full_text = prefix + continuation
    prefix_char_len = len(prefix)

    enc = tokenizer(
        full_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    offsets = enc["offset_mapping"][0].tolist()
    ids = input_ids[0].tolist()

    cont_positions = []
    for tok_idx, (start, end) in enumerate(offsets):
        if end <= prefix_char_len:
            continue
        if start == end:
            continue
        cont_positions.append(tok_idx)

    if not cont_positions:
        return 0.0

    logits = model(input_ids).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)

    total = 0.0
    for tok_idx in cont_positions:
        if tok_idx == 0:
            continue
        total += float(log_probs[tok_idx - 1, ids[tok_idx]].item())

    return total
