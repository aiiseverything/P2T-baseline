#!/usr/bin/env bash
# Shared defaults only; sourcing this file never launches training.
default_initial_adapter() {
  local model_name="${1%/}"
  case "${model_name##*/}" in
    Qwen3-14B-Base) printf '%s\n' 'models/sft-native-eos-clean2k5e2' ;;
    *) echo "INIT_ADAPTER is required for $1; the default native adapter is for Qwen3-14B-Base only" >&2; return 2 ;;
  esac
}

default_sft_output() {
  local model_name="${1%/}"
  case "${model_name##*/}" in
    Qwen3-14B-Base) printf '%s\n' 'models/sft-native-eos-qwen3-14b-base' ;;
    Qwen3-8B-Base) printf '%s\n' 'models/sft-native-eos-qwen3-8b-base' ;;
    *) echo "SFT_OUTPUT is required for model $1" >&2; return 2 ;;
  esac
}
