#!/usr/bin/env bash
# Fan out the AlpacaEval generation sweep over the GPUs of one rjob:
# ALPACA_FANOUT independent single-GPU vLLM engines (default 3), each sweeping
# its shard of the tags given via ALPACA_ADAPTERS.
# Near-linear speedup; per-tag resume still works.
set -uo pipefail
R=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
cd "$R"
N="${ALPACA_FANOUT:-3}"

pids=()
for k in $(seq 0 $((N - 1))); do
  ALPACA_SHARD=$k/$N bash "$R/scripts/run_alpaca.sh" > "/tmp/alpaca_shard$k.log" 2>&1 &
  pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "SHARD $i FAILED — last lines:"; tail -5 "/tmp/alpaca_shard$i.log"
    rc=1
  fi
done
echo "=== per-shard tails ==="
for k in $(seq 0 $((N - 1))); do echo "--- shard $k:"; tail -3 "/tmp/alpaca_shard$k.log"; done
exit $rc
