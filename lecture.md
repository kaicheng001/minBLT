# from bytes to patch predictions

A byte language model assigns `p(x) = product_t p(x[t] | x[:t])`. The alphabet
has 256 symbols even when text contains characters represented by several UTF-8
bytes. BLT changes where most computation happens; it keeps this likelihood.
The [archival BLT paper](https://aclanthology.org/2025.acl-long.453/) describes
the architecture, and the [README](README.md) records the inspected code revision.

## uncertainty chooses boundaries

The separate [entropy model](minblt/entropy.py) trains with next-byte cross-entropy.
At target position `t`, its shifted inputs end at `x[t-1]`. If its distribution is
`p_t`, its entropy is `H[t] = -sum_b p_t[b] log(p_t[b])`. This measures uncertainty
before observing `x[t]`, whereas surprisal measures the probability of that byte
after choosing it. Uniform predictions have entropy `ln(256)` nats.

[patch.py](minblt/patch.py) starts a new payload patch at high `H[t]` or a large
increase `H[t]-H[t-1]`. Meta's relative helper forces its first candidate. A length
cap prevents unbounded spans. Once calibrated on training bytes, the threshold
stays fixed: selecting the largest entropies of a whole continuation would use
future information. Hard boundaries are preprocessing decisions, not a
differentiable path back into the frozen entropy model.

## bytes become patch states

The encoder in [blt.py](minblt/blt.py) adds causal polynomial n-gram embeddings
to byte embeddings. Leading positions include zero padding and BOS; signed-int64
overflow occurs before reducing to a table index. Hash collisions intentionally
share vectors. These features do not create a BPE vocabulary.

Each encoder block applies local causal attention and SwiGLU. RoPE rotates each
query/key coordinate pair by an angle determined by position and frequency;
cross-attention uses no positional rotation. See [base.py](minblt/base.py).

After the first encoder block, each patch mean is projected into K distinct
queries. Queries attend only to that patch's byte states, with pre-RMSNorm and a
residual. Subsequent encoder layers refine the same queries. Concatenating K
query states and projecting them produces one wider global input per patch.
The global causal Transformer can therefore spend more capacity at fewer
positions than a Transformer of the same width running on every byte.

## a completed patch conditions the next one

The critical index is the *target* byte. For targets `x[0:T]`, encode
`z = [BOS, x[0], ..., x[T-1]]`. Put BOS in its own patch. If payload patch lengths
are `[3,2,2]`, encoder patch lengths are `[1,3,2,2]`.

Decoder position `t` receives the encoder state of `z[t]`, so it can know BOS or
`x[t-1]`. Removing the initial encoder patch from the length list gives decoder
IDs `[0,0,0,1,1,2,2]`. Global state 0 summarizes BOS and conditions payload patch
0. Global state 1 summarizes the completed payload patch 0 and conditions patch
1. A patch summary can include all its bytes because it is hidden until those
bytes have been observed.

The decoder projects each global state into K keys/values. Its default mask
selects exactly the aligned global patch; a finite history window can include
earlier patches. Every decoder layer applies cross-attention before local causal
self-attention. The final RMSNorm and 256-way head predict individual bytes.

## train and sample the same distribution

[train.py](train.py) trains the entropy model, freezes it, and trains BLT on the
same chronological training split. Mean NLL counts valid bytes only, and
`BPB = total NLL / (valid bytes * ln(2))`. The held-out subset resets BOS per
window; it is a small reproducible measurement, not a paper benchmark.

Generation runs the same model on a prefix plus an arbitrary dummy target,
samples its final 256-way distribution, and appends that raw byte. Causal masks
must make the dummy's value irrelevant. The [tests](tests/test_minblt.py) challenge
this with suffix mutations, independent masks, scalar mathematical oracles,
gradient checks and teacher-forced versus prefix-only comparisons. A checkpoint
round trip preserves the entropy model, threshold and architecture as well as
the BLT weights, so reloading preserves the distribution being sampled.
