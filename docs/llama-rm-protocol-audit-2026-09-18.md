# Llama actor and Skywork reward protocol audit

The Llama reward serialization and attribution fixes are covered by real saved-tokenizer regressions. CPU checks do not establish that the 8B GPU computation passes: the experiment launcher must also require a successful GPU protocol report and the separate training preflight.

## Assets and runtime

| Role | Local asset |
| --- | --- |
| Base actor | `/data/VPO-RM/models/Llama-3.1-8B-Instruct` |
| Initial SFT adapter and actor tokenizer | `/data/VPO-RM/models/sft-llama31-8b-instruct-clean2k5e2-20260917` |
| Reward model | `/data/VPO-RM/models/Skywork-Reward-Llama-3.1-8B-v0.2` |

The RM is `LlamaForSequenceClassification`: vocabulary 128256, hidden size 4096, 32 decoder layers, BF16 weights. Its index contains 291 tensors totaling 15,009,857,536 tensor bytes. `score.weight` is a finite `[1, 4096]` tensor; there is no score bias or separate reward-normalization tensor. The embedding table is `[128256, 4096]`.

CPU checks first used the existing `sml` environment (Transformers 4.57.6, PyTorch 2.11.0+cu128), then were repeated with the installed Transformers 5.16.1/tokenizers 0.23.2 packages on the same CPU PyTorch environment. The latter full regression passed 950 tests with 4 skips. Transformers 4 cannot resolve the saved SFT export's `TokenizersBackend` class through `AutoTokenizer`. The checker explicitly loads the **saved artifacts** using `PreTrainedTokenizerFast`; it does not substitute a base tokenizer. The formal GPU runtime is separately pinned by the launcher. A CPU pass is never accepted as a GPU pass.

The first GPU allocation exposed an API-default mismatch before model loading: Transformers 5.16.1 defaults `apply_chat_template` to `return_dict=True`, whereas Transformers 4 returns a token list by default. Iterating the new `BatchEncoding` returned field names, causing a false canonical-ID mismatch. Every token-list comparison in the checker now explicitly requests `return_dict=False`. A regression reproduces the dictionary default against a real tokenizer, and both installed Transformers versions are checked on CPU. The failed frozen source/report remains historical evidence; repaired sources require a fresh freeze and GPU gate.

## Native protocol

| Meaning | ID |
| --- | ---: |
| BOS `<\|begin_of_text\|>` | 128000 |
| End of text | 128001 |
| Dedicated pad | 128004 |
| Header start / end | 128006 / 128007 |
| End of message | 128008 |
| End of assistant turn, EOT | 128009 |

The native stop set is `[128001, 128008, 128009]`. Actor and RM must keep their dedicated padding token, 128004. The local base, SFT, and RM templates insert the same automatic system text, including `Cutting Knowledge Date: December 2023` and fixed `Today Date: 26 Jul 2024`. They trim message content. Token vocabulary/backend identity is verified independently of chat-template identity: the RM owns the scoring conversation even if the actor template differs.

Reward input is the complete RM user/assistant conversation, with exactly one initial BOS and a final assistant EOT. Rendered template text is tokenized with `add_special_tokens=False`, preserving exactly the IDs from `apply_chat_template(tokenize=True)`. The published Skywork example scores the native classifier's raw scalar logit, without a sigmoid or reward normalization. Its sample preferred/rejected scores are 13.6875 and -9.1875; numerical agreement with those historical values is not used as a cross-runtime tolerance requirement. [Skywork model card](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2), [HF chat templating guidance](https://huggingface.co/docs/transformers/v4.57.1/chat_templating)

## Reproduced defects and corrections

1. **BOS broke every Llama attribution span.** Previously `_render` stripped BOS, while tokenization inserted it again. Canonical score IDs were correct, but their reconstructed bytes could never equal the stripped rendered text. `Hello world!` plus EOT, `[9906, 1917, 0, 128009]`, mapped to `[-1, -1, -1, -1]`. Preserving the complete rendered text maps it to `[37, 38, 39, -1]` for the `Say hello` probe.
2. **Trailing/leading whitespace trimming discarded surviving content.** The previous mapping only supported removing leading newlines. Llama also removes trailing whitespace and leading spaces/tabs. `Hello\n` therefore lost even the unchanged `Hello` gradient. The corrected path accepts an exact contiguous body only when removed prefix/suffix characters are whitespace. Only whole tokens with identical IDs and byte boundaries map. A token containing a removed leading space remains unmapped; separate unchanged tokens still map. Empty or ambiguously shortened all-whitespace bodies receive no invented positions.
3. **Tokenizer attributes omitted backend special tokens.** In the CPU runtime, `all_special_ids` lists only BOS/EOT/PAD although the backend removes additional registered specials during decoding. An end-of-text or EOM token could therefore invalidate the entire reconstruction. The shared `get_special_token_ids` helper includes backend `AddedToken.special` metadata; alignment and trainer content statistics now agree with decoding.

Unmapped response tokens retain fixed credit one; their sequence advantage remains intact. The actor's emitted stop is removed from decoded response text and gets no fabricated RM gradient. The RM's own final EOT remains part of the score input and is the pooling token. Unicode byte offsets, rather than character offsets or text searching for repeated words, establish token identity.

## Reward pooling and gradients

`LastTokenReward` applies the scalar head at the last **attended** position. Its mask-derived position IDs support both padding sides. Native HF classifier reference scoring must preserve `config.pad_token_id=128004`: changing it to EOT would cause native classifier pooling to skip the real final assistant EOT. Native classifiers receiving only `inputs_embeds` cannot infer padding from token IDs; this is why the wrapper pools explicitly. [Transformers 4.57.6 classifier implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/modeling_layers.py)

Input-gradient calculation freezes all RM parameters, enters evaluation mode, and differentiates the sum of independent response rewards with respect to detached input embeddings. It must produce finite nonzero mapped-content gradients, zero masked-pad gradients, and no parameter gradients or parameter changes. No batch-mean scaling is applied to the reward gradients.

The RM tokenizer advertises a 4096-token limit; the architecture advertises 131072 positions. This experiment uses a 4096-token RM budget. Serialization never silently truncates. The checker constructs real 4096/4097-token native conversations, verifies exact official IDs, accepts the former and rejects the latter. The trainer also rejects actual canonical rows exceeding the smaller of its configured prompt-plus-response budget and the RM tokenizer limit, before embedding lookup; it records the observed maximum RM length and its budget.

This runtime guard is necessary even when actor tokens fit: Llama token 2275 decodes individually to `о�`, which re-encodes as two tokens `[1482, 5809]`. A real-tokenizer regression with an 1835-token actor prefix and a 2048-token capped response produces 5932 canonical RM tokens. The row is rejected without truncation. Actual long-sequence GPU capacity remains a separate launch gate.

## Executable launch gate

```bash
python scripts/check_llama_protocol.py \
  --actor /data/VPO-RM/models/Llama-3.1-8B-Instruct \
  --reward /data/VPO-RM/models/Skywork-Reward-Llama-3.1-8B-v0.2 \
  --init-adapter /data/VPO-RM/models/sft-llama31-8b-instruct-clean2k5e2-20260917 \
  --output /path/to/protocol-cpu.json
```

Add `--gpu` to load only the actual reward model on `cuda:0`. The production BF16 phase compares native-classifier, wrapper-forward and wrapper-input-gradient scores on identical singleton and padded inputs, at absolute tolerance 0.125 and zero relative tolerance. Cross-layout BF16 differences are recorded separately. A second validation-only phase casts the entire same RM, including its backbone and head, to FP32, disables TF32, and requires both same-input equivalence and singleton/left/right-padding invariance within 0.001. Both phases require correct final-EOT pooling, finite nonzero mapped input gradients, zero padding gradients, and unchanged frozen parameters. The FP32 audit model exits with the checker process; calibration and training separately load the original BF16 checkpoint. Actor loading, PEFT update/reference isolation, rollout probabilities, and long-sequence GPU capacity remain separate formal preflight requirements.

This distinction follows the actual H200 diagnostic at `/data/VPO-RM/.maintenance/llama-rl-20260918/rm-padding-diagnostic-v2`: all 84 forward cases completed, with 288 comparisons and no execution errors. Native and wrapper BF16 scores and pooled hidden states were exactly equal on the same inputs, as were no-gradient and input-gradient paths. The original failure was a cross-layout comparison: native SDPA scored the ASCII singleton -3.5625 and its right-padded batch row -3.40625, a difference of 0.15625. Both project paths reproduced the same native values. Casting only the head to FP32 retained the approximately 0.159917 difference, locating it in the BF16 backbone computation. Switching to eager attention also changed native scores and is not used as a protocol correction.

The new full-FP32 backbone control must pass on the real GPU before training is admitted; the prior head-only diagnostic does not establish that result. The profile now checks both the resolved configuration and the actual loaded trainer configuration for physical `microbatch_responses=1`, covering calibration, preflight and formal training. Production reward arithmetic and model weights remain unchanged.

Reports use `schema=llama_protocol_v1`, `status=passed|failed`, and `mode=cpu|gpu`. They include resolved actor/reward/adapter paths, actual SHA256 hashes of metadata/tokenizers/templates/indices and saved adapter weights, and hashes of the checker plus relevant protocol/trainer sources. The launcher must bind CPU and GPU reports to the same identity and frozen source hashes. Exceptions write a failed report and exit nonzero.

The checker verifies checkpoint shard sizes, BF16 tensor headers, tensor-index membership, and small score-head contents. Large shard SHA256 values in `checkpoint_files` are explicitly labeled as prior download evidence, not fresh hashes. Full shard revalidation is recorded separately in `/data/VPO-RM/.maintenance/llama-rl-20260918/assets_verified.json`; launch identity must preserve that evidence.

Regression coverage is in `tests/test_llama_reward_protocol.py` and `tests/test_check_llama_protocol.py`, supplemented by existing Qwen, reward-wrapper, fixed-credit, and microbatch tests. Real-tokenizer tests cover ASCII, Chinese/emoji, whitespace, repeated text, marker collisions, empty answers, all native stops, backend structural specials, differing actor/RM templates, and length boundaries. The checker scoring path is exercised with a small real Llama model on CPU before any full-weight GPU run.
