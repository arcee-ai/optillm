import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Dict, Tuple

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def entropy_spike_beam(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    top_k: int = 50,
    temperature: float = 1.0,
    beam_width: int = 5,
) -> Tuple[str, float]:
    """
    Implement Entropy-Spike Beam Search for a given chat input.

    Args:
        model: The Hugging Face transformer model.
        tokenizer: The associated tokenizer.
        messages: List of chat messages in the format [{"role": "user", "content": "..."}]
        max_new_tokens: Maximum number of new tokens to generate.
        top_k: The number of top tokens to consider at each step.
        temperature: Sampling temperature.
        beam_width: The beam width (number of beams to keep).

    Returns:
        A tuple containing the generated text and its cumulative entropy score.
    """
    device = get_device()
    model.to(device)

    # Use the chat template to format the input
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback for tokenizers without chat templates
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        input_text += "\nassistant:"

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Set pad_token_id if it's not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initialize beams with the input prompt
    beams = [(input_ids, 0.0, [])]  # (input_ids, cumulative_entropy, tokens_generated)

    for _ in range(max_new_tokens):
        all_candidates = []
        for beam_input_ids, beam_entropy, beam_tokens in beams:
            with torch.no_grad():
                outputs = model(beam_input_ids)
                next_token_logits = outputs.logits[:, -1, :]

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Get top_k token probabilities
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                top_k_probs = torch.softmax(top_k_logits, dim=-1)

                # Sample from top_k
                for _ in range(beam_width):
                    next_token = torch.multinomial(top_k_probs, num_samples=1)
                    next_token_idx = next_token.item()
                    next_token_id = top_k_indices[0, next_token_idx].unsqueeze(0)

                    new_input_ids = torch.cat([beam_input_ids, next_token_id.unsqueeze(0)], dim=-1)

                    # Calculate entropy for this new token
                    token_prob = top_k_probs[0, next_token_idx].item()
                    if token_prob > 0:
                        token_entropy = -token_prob * torch.log2(torch.tensor(token_prob)).item()
                    else:
                        token_entropy = 0.0  # Avoid log(0)

                    new_cumulative_entropy = beam_entropy + token_entropy
                    new_tokens = beam_tokens + [next_token_id.item()]

                    all_candidates.append((new_input_ids, new_cumulative_entropy, new_tokens))

        # Select top beam_width candidates based on lowest cumulative entropy
        beams = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

        # Check if any beam has generated an EOS token
        for beam_input_ids, beam_entropy, beam_tokens in beams:
            if tokenizer.eos_token_id in beam_tokens[-1:]:
                generated_tokens = beam_tokens
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                return generated_text, beam_entropy

    # If max_new_tokens is reached, return the text from the beam with lowest entropy
    best_beam = min(beams, key=lambda x: x[1])
    generated_tokens = best_beam[2]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text, best_beam[1]
