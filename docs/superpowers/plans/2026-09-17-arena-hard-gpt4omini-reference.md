# Arena-Hard v2 against GPT-4o-mini

Reuse the six canonical policies' existing hard500 answers byte-for-byte. Compare
them against the official published `gpt-4o-mini-2024-07-18` responses for exactly
those same question UIDs and prompts. Keep GPT-4.1, temperature 0, max 16000 judge
tokens, both answer orders, official raw and combined length/Markdown scoring.
This is a custom-reference evaluation, distinct from the default o3-mini results.

1. Pin official reference source revision and verify prompt and metadata coverage.
2. Add explicit baseline identity to judge/scorer with default-protocol regression
   tests; forbid reuse of judgments produced against a different reference.
3. Freeze a separate judge-only suite, original answer hashes, original generation
   manifests, and the predeclared uniform structural-invalid retry policy.
4. Test and independently review the suite offline; dry-run all 6000 requests.
5. Run a 60-game pilot, then 32 concurrent requests with the existing measured-cost
   dispatch guard. Never silently retry ambiguous transport or drop an invalid row.
6. Aggregate only after all 6000 games are valid, independently reproduce official
   numerical scores, and persist completion plus raw/style win rates and 90% CIs.

No GPU jobs, model inference, model weight changes, or old-result changes are needed.
