from typing import Tuple


def compute_num_lora_params(
    num_encoder_layers: int,
    num_decoder_layers: int,
    emb_dim: int,
    lora_ranks: Tuple,
    num_heads: int,
    head_dim: int,
) -> int:
    total_layers = num_encoder_layers + 2 * num_decoder_layers

    q_rank = lora_ranks[0] or 0
    q_A = total_layers * emb_dim * q_rank
    q_B = total_layers * q_rank * num_heads * head_dim

    k_rank = lora_ranks[1] or 0
    k_A = total_layers * emb_dim * k_rank
    k_B = total_layers * k_rank * num_heads * head_dim

    v_rank = lora_ranks[2] or 0
    v_A = total_layers * emb_dim * v_rank
    v_B = total_layers * v_rank * num_heads * head_dim

    o_rank = lora_ranks[3] or 0
    o_A = total_layers * num_heads * head_dim * o_rank
    o_B = total_layers * o_rank * emb_dim

    lora_A_params = q_A + k_A + v_A + o_A
    lora_B_params = q_B + k_B + v_B + o_B

    print(f"Total lora A parameters: {lora_A_params}")
    print(f"Total lora B parameters: {lora_B_params}")
    print(f"Total lora parameters: {lora_A_params + lora_B_params}")

    return lora_A_params + lora_B_params


if __name__ == "__main__":
    pass
