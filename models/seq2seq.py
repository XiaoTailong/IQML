from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from iqml.quantum.circuits import QuantumCircuitConfig, parameter_count, run_pqc_batch

Params = dict[str, Any]


@dataclass(frozen=True)
class Seq2SeqConfig:
    source_vocab_size: int
    target_vocab_size: int
    embedding_dim: int = 32
    hidden_dim: int = 64
    method: str = "lstm"
    num_qubits: int = 8
    circuit_type: str = "he"
    fusion_hidden_dim: int = 64
    entanglement: str = "linear"
    context_scale: float = 1.0
    mlp_context_hidden_dim: int = 16


@dataclass(frozen=True)
class Seq2SeqOutput:
    logits: jnp.ndarray
    encoder_hidden: jnp.ndarray
    quantum_features: jnp.ndarray | None
    theta: jnp.ndarray | None
    context_features: jnp.ndarray | None
    encoder_quantum_features: jnp.ndarray | None
    decoder_quantum_features: jnp.ndarray | None
    encoder_theta: jnp.ndarray | None
    decoder_theta: jnp.ndarray | None
    decoder_context_features: jnp.ndarray | None


def init_seq2seq_params(key: jax.Array, config: Seq2SeqConfig) -> Params:
    keys = jax.random.split(key, 10)
    decoder_input_dim = config.embedding_dim + config.hidden_dim + _context_dim(config)
    output_input_dim = config.hidden_dim + _context_dim(config)
    params: Params = {
        "source_embedding": _normal(keys[0], (config.source_vocab_size, config.embedding_dim), 0.05),
        "target_embedding": _normal(keys[1], (config.target_vocab_size, config.embedding_dim), 0.05),
        "encoder": _init_lstm(keys[2], config.embedding_dim, config.hidden_dim),
        "decoder": _init_lstm(keys[3], decoder_input_dim, config.hidden_dim),
        "decoder_init": {
            "W_h": _glorot(keys[4], (config.hidden_dim, config.hidden_dim)),
            "b_h": jnp.zeros((config.hidden_dim,), dtype=jnp.float32),
            "W_c": _glorot(keys[5], (config.hidden_dim, config.hidden_dim)),
            "b_c": jnp.zeros((config.hidden_dim,), dtype=jnp.float32),
        },
        "output": {
            "W": _glorot(keys[6], (output_input_dim, config.target_vocab_size)),
            "b": jnp.zeros((config.target_vocab_size,), dtype=jnp.float32),
        },
    }
    if config.method.lower() in {"mlp_context", "small_mlp_context"}:
        params["context_mlp"] = {
            "W1": _glorot(keys[7], (config.hidden_dim, config.mlp_context_hidden_dim)),
            "b1": jnp.zeros((config.mlp_context_hidden_dim,), dtype=jnp.float32),
            "W2": _glorot(keys[8], (config.mlp_context_hidden_dim, config.num_qubits)),
            "b2": jnp.zeros((config.num_qubits,), dtype=jnp.float32),
        }
    return params


def seq2seq_forward(
    params: Params,
    source: jnp.ndarray,
    decoder_input: jnp.ndarray,
    config: Seq2SeqConfig,
) -> Seq2SeqOutput:
    source_emb = params["source_embedding"][source]
    encoder_hidden = _lstm_sequence(params["encoder"], source_emb, config.hidden_dim)
    context = encoder_hidden[:, -1, :]
    quantum_features = None
    theta = None
    encoder_quantum_features = None
    decoder_quantum_features = None
    encoder_theta = None
    decoder_theta = None
    decoder_context_features = None
    context_features = _context_features(encoder_hidden, config)

    if config.method.lower().startswith("iqml"):
        encoder_theta = _hidden_to_theta(encoder_hidden, config)
        encoder_quantum_features = run_pqc_batch(encoder_theta, _quantum_config(config))
        theta = encoder_theta
        quantum_features = encoder_quantum_features
        context_features = encoder_quantum_features
    elif config.method.lower() in {"mlp_context", "small_mlp_context"}:
        context_features = _mlp_context_features(params["context_mlp"], encoder_hidden[:, -1, :])

    initial_h = jnp.tanh(context @ params["decoder_init"]["W_h"] + params["decoder_init"]["b_h"])
    initial_c = jnp.tanh(context @ params["decoder_init"]["W_c"] + params["decoder_init"]["b_c"])
    target_emb = params["target_embedding"][decoder_input]
    repeated_context = jnp.repeat(context[:, None, :], target_emb.shape[1], axis=1)
    decoder_inputs = [target_emb, repeated_context]
    if context_features is not None:
        repeated_features = jnp.repeat(context_features[:, None, :], target_emb.shape[1], axis=1)
        decoder_inputs.append(repeated_features)
    decoder_input_features = jnp.concatenate(decoder_inputs, axis=-1)
    decoder_hidden = _lstm_sequence(
        params["decoder"],
        decoder_input_features,
        config.hidden_dim,
        initial_state=(initial_h, initial_c),
    )
    if config.method.lower().startswith("iqml"):
        decoder_theta = _hidden_to_theta(decoder_hidden, config)
        decoder_quantum_features = _run_stepwise_pqc(decoder_theta, _quantum_config(config))
        decoder_context_features = decoder_quantum_features
    elif config.method.lower() in {"classical_context", "lstm_context"}:
        decoder_context_features = _decoder_context_features(decoder_hidden, config)
    elif config.method.lower() in {"mlp_context", "small_mlp_context"}:
        decoder_context_features = _mlp_context_features(params["context_mlp"], decoder_hidden)

    output_inputs = [decoder_hidden]
    if decoder_context_features is not None:
        output_inputs.append(decoder_context_features)
    output_features = jnp.concatenate(output_inputs, axis=-1)
    logits = _linear(params["output"], output_features)
    return Seq2SeqOutput(
        logits=logits,
        encoder_hidden=encoder_hidden,
        quantum_features=quantum_features,
        theta=theta,
        context_features=context_features,
        encoder_quantum_features=encoder_quantum_features,
        decoder_quantum_features=decoder_quantum_features,
        encoder_theta=encoder_theta,
        decoder_theta=decoder_theta,
        decoder_context_features=decoder_context_features,
    )


def seq2seq_encoder_hidden(
    params: Params,
    source: jnp.ndarray,
    config: Seq2SeqConfig,
) -> jnp.ndarray:
    source_emb = params["source_embedding"][source]
    return _lstm_sequence(params["encoder"], source_emb, config.hidden_dim)


def seq2seq_theta_from_hidden(hidden_sequence: jnp.ndarray, config: Seq2SeqConfig) -> jnp.ndarray:
    return _hidden_to_theta(hidden_sequence, config)


def seq2seq_decoder_hidden_from_context_features(
    params: Params,
    encoder_hidden: jnp.ndarray,
    decoder_input: jnp.ndarray,
    context_features: jnp.ndarray | None,
    config: Seq2SeqConfig,
) -> jnp.ndarray:
    context = encoder_hidden[:, -1, :]
    initial_h = jnp.tanh(context @ params["decoder_init"]["W_h"] + params["decoder_init"]["b_h"])
    initial_c = jnp.tanh(context @ params["decoder_init"]["W_c"] + params["decoder_init"]["b_c"])
    target_emb = params["target_embedding"][decoder_input]
    repeated_context = jnp.repeat(context[:, None, :], target_emb.shape[1], axis=1)
    decoder_inputs = [target_emb, repeated_context]
    if context_features is not None:
        repeated_features = jnp.repeat(context_features[:, None, :], target_emb.shape[1], axis=1)
        decoder_inputs.append(repeated_features)
    decoder_input_features = jnp.concatenate(decoder_inputs, axis=-1)
    return _lstm_sequence(
        params["decoder"],
        decoder_input_features,
        config.hidden_dim,
        initial_state=(initial_h, initial_c),
    )


def seq2seq_logits_from_decoder_features(
    params: Params,
    decoder_hidden: jnp.ndarray,
    decoder_context_features: jnp.ndarray | None,
) -> jnp.ndarray:
    output_inputs = [decoder_hidden]
    if decoder_context_features is not None:
        output_inputs.append(decoder_context_features)
    output_features = jnp.concatenate(output_inputs, axis=-1)
    return _linear(params["output"], output_features)


def translation_loss(
    params: Params,
    source: jnp.ndarray,
    decoder_input: jnp.ndarray,
    target: jnp.ndarray,
    target_mask: jnp.ndarray,
    config: Seq2SeqConfig,
) -> jnp.ndarray:
    output = seq2seq_forward(params, source, decoder_input, config)
    return masked_cross_entropy(output.logits, target, target_mask)


def masked_cross_entropy(logits: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_loss = -jnp.take_along_axis(log_probs, target[..., None], axis=-1)[..., 0]
    return jnp.sum(token_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def greedy_decode(
    params: Params,
    source: jnp.ndarray,
    config: Seq2SeqConfig,
    max_target_len: int,
    bos_id: int = 1,
) -> jnp.ndarray:
    batch_size = source.shape[0]
    decoder_tokens = jnp.zeros((batch_size, max_target_len), dtype=jnp.int32)
    decoder_tokens = decoder_tokens.at[:, 0].set(bos_id)
    predictions = jnp.zeros((batch_size, max_target_len), dtype=jnp.int32)

    def body(
        step: int,
        carry: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        current_decoder, current_predictions = carry
        output = seq2seq_forward(params, source, current_decoder, config)
        next_token = jnp.argmax(output.logits[:, step, :], axis=-1).astype(jnp.int32)
        next_predictions = current_predictions.at[:, step].set(next_token)
        next_decoder = jax.lax.cond(
            step + 1 < max_target_len,
            lambda value: value.at[:, step + 1].set(next_token),
            lambda value: value,
            current_decoder,
        )
        return next_decoder, next_predictions

    _, predictions = jax.lax.fori_loop(0, max_target_len, body, (decoder_tokens, predictions))
    return predictions


def _init_lstm(key: jax.Array, input_dim: int, hidden_dim: int) -> Params:
    w_key, b_key = jax.random.split(key)
    del b_key
    return {
        "W": _glorot(w_key, (input_dim + hidden_dim, 4 * hidden_dim)),
        "b": jnp.zeros((4 * hidden_dim,), dtype=jnp.float32),
    }


def _lstm_sequence(
    params: Params,
    x: jnp.ndarray,
    hidden_dim: int,
    initial_state: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    batch_size = x.shape[0]
    if initial_state is None:
        h0 = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
        c0 = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
    else:
        h0, c0 = initial_state

    def step(carry: tuple[jnp.ndarray, jnp.ndarray], x_t: jnp.ndarray):
        h, c = carry
        joined = jnp.concatenate([x_t, h], axis=-1)
        gates = joined @ params["W"] + params["b"]
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        next_c = f * c + i * g
        next_h = o * jnp.tanh(next_c)
        return (next_h, next_c), next_h

    (_, _), hidden_time_major = jax.lax.scan(step, (h0, c0), jnp.swapaxes(x, 0, 1))
    return jnp.swapaxes(hidden_time_major, 0, 1)


def _quantum_config(config: Seq2SeqConfig) -> QuantumCircuitConfig:
    return QuantumCircuitConfig(
        num_qubits=config.num_qubits,
        circuit_type=config.circuit_type,
        entanglement=config.entanglement,
    )


def _hidden_to_theta(encoder_hidden: jnp.ndarray, config: Seq2SeqConfig) -> jnp.ndarray:
    parameter_dim = parameter_count(_quantum_config(config))
    normalized = _rms_norm(encoder_hidden)
    tiled = _take_repeated_features(normalized, parameter_dim)
    return jnp.pi * jnp.tanh(config.context_scale * tiled)


def _context_features(encoder_hidden: jnp.ndarray, config: Seq2SeqConfig) -> jnp.ndarray | None:
    method = config.method.lower()
    if method not in {"classical_context", "lstm_context"}:
        return None
    normalized = _rms_norm(encoder_hidden[:, -1, :])
    return _take_repeated_features(normalized, config.num_qubits)


def _decoder_context_features(decoder_hidden: jnp.ndarray, config: Seq2SeqConfig) -> jnp.ndarray:
    normalized = _rms_norm(decoder_hidden)
    return _take_repeated_features(normalized, config.num_qubits)


def _run_stepwise_pqc(theta: jnp.ndarray, config: QuantumCircuitConfig) -> jnp.ndarray:
    seq_len = theta.shape[1]

    def feature_at_step(step: int) -> jnp.ndarray:
        active = jnp.arange(seq_len) <= step
        padded_theta = theta * active[None, :, None]
        return run_pqc_batch(padded_theta, config)

    features_time_major = jax.vmap(feature_at_step)(jnp.arange(seq_len))
    return jnp.swapaxes(features_time_major, 0, 1)


def _context_dim(config: Seq2SeqConfig) -> int:
    method = config.method.lower()
    if method.startswith("iqml") or method in {"classical_context", "lstm_context", "mlp_context", "small_mlp_context"}:
        return config.num_qubits
    return 0


def _mlp_context_features(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    hidden = jax.nn.gelu(_linear({"W": params["W1"], "b": params["b1"]}, _rms_norm(x)))
    return jnp.tanh(_linear({"W": params["W2"], "b": params["b2"]}, hidden))


def _take_repeated_features(x: jnp.ndarray, width: int) -> jnp.ndarray:
    if width <= 0:
        return x[..., :0]
    repeats = (width + x.shape[-1] - 1) // x.shape[-1]
    tiled = jnp.tile(x, (*([1] * (x.ndim - 1)), repeats))
    return tiled[..., :width]


def _rms_norm(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms


def _linear(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    return x @ params["W"] + params["b"]


def _glorot(key: jax.Array, shape: tuple[int, int]) -> jnp.ndarray:
    fan_in, fan_out = shape
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=jnp.float32)


def _normal(key: jax.Array, shape: tuple[int, int], std: float) -> jnp.ndarray:
    return std * jax.random.normal(key, shape, dtype=jnp.float32)
