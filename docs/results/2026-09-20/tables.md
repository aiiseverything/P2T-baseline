main

| Actor | RM | 方法 | RM-Reward | AlapacaEval | IF-Eval | Arena-Hard |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-14B-Base | Skywork-Qwen3-8B | base | -2.537 | 6.65 | 38.07 | 8.42 |
| Qwen3-14B-Base | Skywork-Qwen3-8B | sft | 3.749 | 7.00 | 58.52 | 3.15 |
| Qwen3-14B-Base | Skywork-Qwen3-8B | grpo | 10.870 | 37.39 | 59.47 | 28.49 |
| Qwen3-14B-Base | Skywork-Qwen3-8B | vpo-lambda(4) | 13.353 | 49.57 | 58.28 | 39.31 |
| Qwen3-14B-Instruct | Skywork-Qwen3-8B | base | 11.426 | 63.4 | 88.03 | 66.75 |
| Qwen3-14B-Instruct | Skywork-Qwen3-8B | grpo | 13.751 | 72.8 | 80.56 | 73.60 |
| Qwen3-14B-Instruct | Skywork-Qwen3-8B | vpo-lambda(4) | 14.433 | 73.7 | 82.00 | 73.16 |
| Llama-3.1-8B-Base | Skywork-Llama-3.1-8B-v0.2 | base | -28.06 | 0.00 | 16.4 | 0.03 |
| Llama-3.1-8B-Base | Skywork-Llama-3.1-8B-v0.2 | sft | -8.08 | 4.02 | 42.23 | 1.24 |
| Llama-3.1-8B-Base | Skywork-Llama-3.1-8B-v0.2 | grpo | 4.55 | 9.73 | 37.65 | 2.83 |
| Llama-3.1-8B-Base | Skywork-Llama-3.1-8B-v0.2 | vpo-lambda(4) | 11.8 | 12.58 | 42.18 | 5.10 |
| Llama-3.1-8B-Instruct | Skywork-Llama-3.1-8B-v0.2 | base | -0.262 | 11.47 | 71.06 | 3.01 |
| Llama-3.1-8B-Instruct | Skywork-Llama-3.1-8B-v0.2 | grpo | 19.608 | 21.61 | 65.43 | 8.41 |
| Llama-3.1-8B-Instruct | Skywork-Llama-3.1-8B-v0.2 | vpo-lambda(4) | 24.726 | 27.08 | 75.82 | 9.35 |

ablation

| Actor | RM | 方法 | RM-Reward | AlapacaEval | IF-Eval | Arena-Hard |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3-14b-base | Skywork-qwen3-8b | base | -2.537 | 6.65 | 38.07 | 8.42 |
| qwen3-14b-base | Skywork-qwen3-8b | sft | 3.749 | 7.00 | 58.52 | 3.15 |
| qwen3-14b-base | Skywork-qwen3-8b | grpo | 10.870 | 37.39 | 59.47 | 28.49 |
| qwen3-14b-base | Skywork-qwen3-8b | vpo-lambda(2) | 12.885 | 41.49 | 61.17 | 34.49 |
| qwen3-14b-base | Skywork-qwen3-8b | vpo-lambda(4) | 13.353 | 49.57 | 58.28 | 39.31 |
| qwen3-14b-base | Skywork-qwen3-8b | vpo-lambda(8) | 12.911 | 45.96 | 53.46 | 37.63 |
| qwen3-14b-base | Skywork-qwen3-8b | vpo-lambda(4)-shuffle | 10.73 | 32.62 | 51.70 | 26.91 |

Main tables use generation seed42. Qwen-Instruct seed43 is reported separately in summary.json.
Raw pooled two-order Arena verdicts, decisive weight3, tie0.5; GPT-4o-mini reference, GPT-4o / GPT-4.1 judges.
Original shuffle row actually denotes random_direction; non-Arena cells preserved verbatim.
Coverage: {"qb": 748, "lb": 749, "li": 749, "qi42": 748, "qi43": 748}
Missing judgments: 8. Do not impute missing verdicts.


本次两轮补测已补回152/160个缺失判定。尚缺8次：6次HTTP503、1次content_filter、1次达到16000 token上限。保持原模型回答、裁判提示词和token预算；缺判不填输赢平局。

两张主表统一使用生成seed42；seed43仅作单独重复评测。Qwen-Base两表共享七模型共同有效题集748/750（hard499+creative249）；Llama-Base749/750（hard500+creative249）；Llama-Instruct749/750（hard500+creative249）；Qwen-Instruct seed42/43均748/750（hard500+creative248）。

原shuffle标签实际对应Random-direction credit消融，正式展示时应改名。其他列按用户原表保留。
