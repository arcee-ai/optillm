import ast
import asyncclick as click
import torch
from openai import AsyncOpenAI
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Tuple


async def generate_one_token(
        client: AsyncOpenAI, 
        model_name: str, 
        prompt: str, 
        temperature: float, 
        top_k: int,
    ):
    response = await client.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=temperature,
        max_tokens=1,
        logprobs=top_k,
    )
    return response


async def entropy_spike_beam(
    model_name: str,
    tokenizer_name: str,
    messages: str,
    max_new_tokens: int = 512,
    top_k: int = 50,
    temperature: float = 1.0,
    beam_width: int = 5,
    inference_endpoint: Optional[str] = None,
) -> Tuple[str, float]:
    """
    Implement Entropy-Spike Beam Search for a given chat input.

    Args:
        model_name: The inference enginemodel name.
        tokenizer_name: The associated tokenizer name.
        messages: List of chat messages in the format [{"role": "user", "content": "..."}]
        max_new_tokens: Maximum number of new tokens to generate.
        top_k: The number of top tokens to consider at each step.
        temperature: Sampling temperature.
        beam_width: The beam width (number of beams to keep).
        inference_endpoint: The inference endpoint to use, defaults to the arcee research endpoint

    Returns:
        A tuple containing the generated text and its cumulative entropy score.
    """

    client = AsyncOpenAI(
        api_key="no_key",
        base_url=inference_endpoint or "https://models.research.arcee.ai/openai/v1"
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    messages_list: List[Dict[str, str]] = ast.literal_eval(messages)

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback for tokenizers without chat templates
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages_list])
        input_text += "\nassistant:"

    beams = [[input_text, 0.0, []]]  # (input, cumulative_entropy, tokens_generated)

    for token_num in tqdm(range(max_new_tokens), desc = 'Generating new beams for each new token...'):
        all_candidates = []
        for beam_prompt, beam_entropy, beam_tokens in tqdm(beams, desc = 'Sampling new tokens for each new beam...', leave=None):
            output = await generate_one_token(
                client=client,
                model_name=model_name,
                prompt=beam_prompt,
                temperature=temperature,
                top_k=top_k,
            )

            top_logprobs = output.choices[0].logprobs.top_logprobs[0]
            top_tokens = list(top_logprobs.keys())
            top_logits = list(top_logprobs.values())

            top_probs = torch.softmax(torch.FloatTensor(top_logits), dim=-1)

            # Sample from top_probs
            for _ in range(beam_width):
                next_token = torch.multinomial(top_probs, num_samples=1)
                next_token_idx = next_token.item()
                next_token_str = top_tokens[next_token_idx]

                new_prompt = beam_prompt + next_token_str

                # Calculate entropy for this new token
                token_prob = top_probs[next_token_idx].item()
                if token_prob > 0:
                    token_entropy = -token_prob * torch.log2(torch.tensor(token_prob)).item()
                else:
                    token_entropy = 0.0  # Avoid log(0)

                new_cumulative_entropy = beam_entropy + token_entropy
                new_tokens = beam_tokens + [next_token_str]

                all_candidates.append([new_prompt, new_cumulative_entropy, new_tokens])

        # Select top beam_width candidates based on lowest cumulative entropy
        beams = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

        # Check if any beam has generated an EOS token, if so return it
        for beam_prompt, beam_entropy, beam_tokens in beams:
            if beam_tokens[-1] == "":  # for whatever reason this isn't "<|im_end|>"
                print(f'a top {beam_width} beam has a EOS token ("") at token number {token_num}, exiting early')
                return beam_tokens, beam_entropy

    # If max_new_tokens is reached, return the text from the beam with lowest entropy
    print(f"reached max_new_tokens of {max_new_tokens}, exiting")
    best_beam = min(beams, key=lambda x: x[1])
    return best_beam[2], best_beam[1]


@click.command()
@click.option('--model_name', default='supernova-medius', help='The inference engine model name.')
@click.option('--tokenizer_name', default='arcee-ai/SuperNova-Medius', help='The associated tokenizer name.')
@click.option('--messages', required=True, help='List of chat messages in the format [{"role": "user", "content": "..."}]')
@click.option('--max_new_tokens', default=1024, help='Maximum number of new tokens to generate.')
@click.option('--top_k', default=50, help='The number of top tokens to consider at each step.')
@click.option('--temperature', default=1.0, help='Sampling temperature.')
@click.option('--beam_width', default=5, help='The beam width (number of beams to keep).')
@click.option('--inference_endpoint', default='https://models.research.arcee.ai/openai/v1', help='The inference endpoint to use, defaults to the arcee research endpoint')
async def run_entropy_spike_beam(
    model_name: str,
    tokenizer_name: str,
    messages: str,
    max_new_tokens: int = 512,
    top_k: int = 50,
    temperature: float = 1.0,
    beam_width: int = 5,
    inference_endpoint: Optional[str] = None,
) -> Tuple[str, float]:
    best_text, best_entropy = await entropy_spike_beam(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        messages=messages,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        temperature=temperature,
        beam_width=beam_width,
        inference_endpoint=inference_endpoint,
    )

    print("minimum entropy: ", best_entropy)
    print("".join(best_text))

    return best_text, best_entropy

if __name__ == '__main__':
    run_entropy_spike_beam()
