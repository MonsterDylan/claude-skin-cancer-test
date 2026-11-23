#!/bin/bash
# Round 1: run every model that has a key, then score.
# Set ANTHROPIC_API_KEY (and optionally OPENAI_API_KEY, GEMINI_API_KEY) in your environment first.
# Override model names with CLAUDE_MODEL / FABLE_MODEL / GPT_MODEL / GEMINI_MODEL.
set -e
cd "$(dirname "$0")"
for m in claude fable gpt gemini; do
  case $m in claude|fable) k=$ANTHROPIC_API_KEY;; gpt) k=$OPENAI_API_KEY;; gemini) k=$GEMINI_API_KEY;; esac
  if [ -n "$k" ]; then python src/run_model.py $m; else echo "skip $m: no key"; fi
done
python src/score.py > /dev/null && echo "scorecard: results/scorecard.json"
