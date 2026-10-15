#!/usr/bin/env bash
# Push this folder + the instruction-tuning mix + keys to a pod made by create_pod.sh.
#   KEYS_FILE=.env bash push_to_pod.sh <pod-name>
# Optional: STORIES_CORPUS=path/to/stories.jsonl to stage the stories corpus too.
set -euo pipefail
NAME="${1:?usage: push_to_pod.sh <pod-name>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/.pods/$NAME.env"
: "${KEYS_FILE:?set KEYS_FILE to the file defining HF_TOKEN and WANDB_API_KEY}"

# Only the two keys the pod needs — never the whole .env.
KEYS_TMP=$(mktemp); trap 'rm -f "$KEYS_TMP"' EXIT
grep -E '^(export )?(HF_TOKEN|WANDB_API_KEY)=' "$KEYS_FILE" > "$KEYS_TMP"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
ssh -p "$SSH_PORT" "${SSH_OPTS[@]}" "root@$SSH_IP" "mkdir -p /workspace/data"
scp -P "$SSH_PORT" "${SSH_OPTS[@]}" -r "$HERE" "root@$SSH_IP:/root/"
scp -P "$SSH_PORT" "${SSH_OPTS[@]}" "$HERE/../../data/instruct_mix/instruct_mix.jsonl" \
  "root@$SSH_IP:/workspace/data/"
if [ -n "${STORIES_CORPUS:-}" ]; then
  [ -f "$STORIES_CORPUS" ] || { echo "STORIES_CORPUS not found: $STORIES_CORPUS"; exit 1; }
  scp -P "$SSH_PORT" "${SSH_OPTS[@]}" "$STORIES_CORPUS" \
    "root@$SSH_IP:/workspace/data/stories.jsonl"
fi
scp -P "$SSH_PORT" "${SSH_OPTS[@]}" "$KEYS_TMP" "root@$SSH_IP:/root/.keys"
echo "pushed training/fsdp to $NAME ($SSH_IP:$SSH_PORT)"
