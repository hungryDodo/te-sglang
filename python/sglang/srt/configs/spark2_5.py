from typing import Any, Optional

from transformers.configuration_utils import PretrainedConfig

_GGUF_ARCH = "spark2_5"


def spark2_5_config_kwargs_from_gguf(gguf_path: str) -> dict[str, Any]:
    """Reconstruct Spark2_5Config kwargs from a llama.cpp spark2_5 GGUF.

    llama.cpp emits one kv block per attention family: the unsuffixed
    ``rope.*`` keys describe full-attention layers and the ``*_swa`` keys the
    sliding-window layers, which is the reverse assignment of
    ``Spark2_5Config.rope_parameters`` naming (full_attention gets the large
    theta with the small partial rotary factor).
    """
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    meta = {key: field.contents() for key, field in reader.fields.items()}
    shapes = {t.name: tuple(int(x) for x in t.shape) for t in reader.tensors}

    def get(suffix):
        return meta[f"{_GGUF_ARCH}.{suffix}"]

    head_dim = int(get("attention.key_length"))
    swa_pattern = [bool(x) for x in get("attention.sliding_window_pattern")]

    return dict(
        # token_embd is stored [n_embd, n_vocab] in ggml's reversed order.
        vocab_size=shapes["token_embd.weight"][1],
        hidden_size=int(get("embedding_length")),
        intermediate_size=int(get("feed_forward_length")),
        num_hidden_layers=int(get("block_count")),
        num_attention_heads=int(get("attention.head_count")),
        num_key_value_heads=int(get("attention.head_count_kv")),
        head_dim=head_dim,
        rms_norm_eps=float(get("attention.layer_norm_rms_epsilon")),
        max_position_embeddings=int(get("context_length")),
        sliding_window=int(get("attention.sliding_window")),
        layer_types=[
            "sliding_attention" if s else "full_attention" for s in swa_pattern
        ],
        rope_parameters={
            "full_attention": {
                "rope_theta": float(get("rope.freq_base")),
                "partial_rotary_factor": int(get("rope.dimension_count")) / head_dim,
            },
            "sliding_attention": {
                "rope_theta": float(get("rope.freq_base_swa")),
                "partial_rotary_factor": int(get("rope.dimension_count_swa")) / head_dim,
            },
        },
        headwise_attn_output_gate=any(
            n.endswith("attn_gate.weight") for n in shapes
        ),
        tie_word_embeddings="output.weight" not in shapes,
        architectures=["Spark2_5ForCausalLM"],
        dtype="bfloat16",
    )


class Spark2_5Config(PretrainedConfig):
    model_type = "spark2_5"
    architectures = ["Spark2_5ForCausalLM"]

    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6656,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        num_hidden_layers: int = 28,
        head_dim: int = 256,
        headwise_attn_output_gate: bool = True,
        sliding_window: int = 512,
        vocab_size: int = 133120,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 8192,
        rope_parameters: Optional[dict[str, Any]] = None,
        layer_types: list[str] = None,
        tie_word_embeddings: Optional[bool] = None,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.sliding_window = sliding_window
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings

        if layer_types is not None:
            layer_types = layer_types[: self.num_hidden_layers]
        else:
            layer_types = [
                "sliding_attention" if bool((i + 1) % 4) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        self.layer_types = layer_types

        if rope_parameters is not None:
            self.rope_parameters = rope_parameters
        else:
            self.rope_parameters = {
                "full_attention": {
                    "rope_theta": 5000000,
                    "partial_rotary_factor": 0.25,
                },
                "sliding_attention": {
                    "rope_theta": 10000,
                    "partial_rotary_factor": 1.0,
                },
            }
        self.hybrid_layer_pattern = [
            1 if layer_type == "sliding_attention" else 0
            for layer_type in self.layer_types
        ]
        self.is_hybrid_swa = any(self.hybrid_layer_pattern)
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)

    @classmethod
    def from_gguf(cls, gguf_path: str) -> "Spark2_5Config":
        return cls(**spark2_5_config_kwargs_from_gguf(gguf_path))
