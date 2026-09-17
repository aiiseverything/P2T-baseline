# Resumed judging through17891

The active controller is `continue_resilient.py`, launched by `launch_resilient.py`.
Read `resilient_status.json` for current status. The old `continuation_state.json`
and `manual_recovery_state.json` intentionally preserve the earlier failed runs.

The fixed500 questions,3000 candidate answers, GPT-4o-mini reference, GPT-4.1
two-order judging protocol, and official scorer remain unchanged. All67 frozen
experiment files and the previous22 valid game files were checked before launch.

`resilient_policy.json` binds the new source files, the22 valid records, the exact
two pre-existing response-free connection failures, the earlier one-game manual
recovery, and the original cumulative billing origin. It permits at most four
connection attempts only when the recorded transport trace proves no application
request headers were sent. Each game allows at most five logical attempts for
eligible connection failures or malformed completed judgments; the first valid
judgment is retained. Uncertain paid outcomes stop the run.

`job/connect_attempts.jsonl` and `job/transport_bindings/` retain transport evidence.
The pilot still uses60 games with12 workers and the existing20CNY dispatch guard.
`resilient_cost_decision.json` records its measured cost and the original budget
formula before full judging with32 workers. There is no billing-baseline reset.

Validation before this launch:113 offline tests passed, including6000 synthetic
saved games through the actual frozen scorer and independent42-value comparison.
This is validation of the pipeline, not completed experimental results.
`evaluation_complete.json` is written only after all6000 actual games and final
independent numerical verification pass.

The live process explicitly inherits HTTP(S) proxy `http://127.0.0.1:17891` in
both uppercase and lowercase variables. At21:37 HKT, a free32-concurrent probe
reached the API in all32 requests (expected401 without credentials).
