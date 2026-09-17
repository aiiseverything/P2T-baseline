# Arena-Hard v2: GPT-4o-mini reference

Six canonical models, same 500 hard prompts and 3000 previously generated
answers. Original answer and generation-manifest bytes are copied unchanged
from `../arena-hard-v2-canonical-20260917`. No GPU rollout is run.

Reference: official published `gpt-4o-mini-2024-07-18`, HF dataset revision
`15f3746e21432264ce9b453999bde4f3c946d2e6`. See `reference/provenance.json`
and `reference/audit.json`. All 500 prompt identities and all answer style
metadata were checked.

Judge: GPT-4.1, two answer orders, temperature 0, max 16000 output tokens.
60-game fixed pilot with 12 workers, then 32 concurrent judge requests.
The pilot is part of the final 6000 games and is not regenerated.
The existing pilot-cost guard is retained; see `cost_decision.json` once available.
`retry_policy.json` is declared before paid calls: first valid result, at most
five identical attempts for completed malformed output; ambiguous transport
is not automatically retried.

Results: `scores/results.csv` and `scores/results.json`, once all 6000 games
are valid. Report raw win rate, length/Markdown controlled win rate, and 90%
confidence intervals. A strict independent verifier must pass before
`evaluation_complete.json` is written. This is a custom-reference evaluation;
its scores must be labeled as against GPT-4o-mini, distinct from the default
o3-mini leaderboard and the original suite.

Entrypoint: `/root/.venvs/alpacaeval/bin/python continue_evaluation.py`.
Status: `continuation_state.json`; logs: `job/controller.log` and
`job/continuation/`. Inputs and implementation hashes: `experiment.json`.

## Pilot transport incident

The first batch saved 11 valid games and one `ConnectError` with no local
response. The original controller stopped as designed. A separately reviewed
recovery is restricted to `lam8 / 160e7f5bbfe84ce0 / order 1`, at most one
byte-identical replacement, within the unchanged cumulative 20 CNY pilot
guard. Existing valid games and all frozen inputs remain unchanged.

The additive scripts are `manual_transport_recovery.py`,
`resume_manual_recovery.py`, and `verify_manual_recovery.py`. The last script
checks the exact original failure archive and replacement; all other 5,999
games retain the original strict verifier. Approval here records the agent's
specific incident review under the user's existing evaluation request.
No claim is made that the first remote attempt was unbilled. See
`manual_transport_recovery.json` and `manual_recovery_state.json` when created.

Recovery wrapper runtime: `/root/miniconda3/envs/sml/bin/python -u resume_manual_recovery.py`. Initial reviewed replacement succeeded with exactly one additional request; all eleven previously valid records retained identical hashes.
