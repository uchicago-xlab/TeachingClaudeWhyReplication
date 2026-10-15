#!/bin/zsh
# Terminate one RunPod pod by id and verify it is gone (404 is the only proof).
#   zsh terminate_pod.sh <pod-id>
# RUNPOD_API_KEY is taken from the environment, else from the repo-root .env.
set -u
POD="${1:?pod id}"
K="${RUNPOD_API_KEY:-$(grep -E '^(export )?RUNPOD_API_KEY=' "$(cd "$(dirname "$0")/../.." && pwd)/.env" | tail -1 | sed 's/^export //; s/^RUNPOD_API_KEY=//; s/^"//; s/"$//')}"
[ -n "$K" ] || { echo "RUNPOD_API_KEY not in environment or .env"; exit 1; }
curl -sS -X DELETE "https://rest.runpod.io/v1/pods/$POD" -H "Authorization: Bearer $K"; echo
for i in 1 2 3 4 5 6; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $K" "https://rest.runpod.io/v1/pods/$POD")
  [ "$code" = "404" ] && { echo "pod $POD terminated (404)"; exit 0; }
  sleep 10
done
echo "WARNING: pod $POD still answers ($code) — check the console"; exit 1
