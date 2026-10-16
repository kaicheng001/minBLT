# exercise

Build a Byte Latent Transformer one causal boundary at a time. Use the included
[minbpe corpus](tests/taylorswift.txt); its attribution is in [README.md](README.md).
These exercises are original and concern BLT rather than BPE merge learning.

### Step 1: bytes

Write UTF-8 `encode` and `decode` helpers. Round-trip accented text and emoji.
Then try `bytes(range(256))`: explain why replacement decoding cannot preserve it.
Reserve special input IDs and keep the prediction head at 256 classes.

### Step 2: patches

Implement fixed boundaries and derive positive lengths and byte-to-patch IDs.
Test empty input, a single byte, and a final partial patch. Split a multibyte UTF-8
character across patches and reconstruct the exact original bytes.

### Step 3: entropy

Train a small causal byte Transformer. Predict target `x[t]` from `[BOS] + x[:t]`.
Compute distribution entropy with log-softmax; verify uniform logits give
`ln(256)`. Explain why `-log p(x[t])` is a different quantity.

### Step 4: dynamic boundaries

Freeze the predictor. Implement global and relative thresholds, including the
reference relative policy's forced first candidate. Add a causal length cap and
calibrate a global threshold on training bytes. Compare full-sequence and online
boundaries. Investigate scores exactly equal to the threshold.

### Step 5: local / global / local

Implement RMSNorm, RoPE, SwiGLU and causal attention. Add left-context hash
embeddings and check signed overflow against Python integers. Project pooled
patch states into K distinct queries, refine them by cross-attention after each
encoder block, and concatenate them for a wider global patch Transformer.

### Step 6: lag correction

Prepend a singleton BOS encoder patch. For payload lengths `[3,2,2]`, derive
encoder lengths `[1,3,2,2]` and decoder IDs `[0,0,0,1,1,2,2]` by hand. Split each
global output into K decoder keys/values. Build the equality and two-patch-history
masks. Prove that no decoder query can read the unfinished current patch.

### Step 7: challenge the model

Mutate a future byte inside a patch and compare earlier logits. Repeat across
patch boundaries, with padding, and with changed future entropy boundaries.
Compare teacher forcing to prefix-only inference and vary the dummy target.
Verify nonzero gradients in all branches, then measure fixed versus dynamic
patch lengths and valid-byte BPB. Sampling fluent prose is a later experiment.
