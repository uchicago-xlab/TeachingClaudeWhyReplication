#!/bin/bash
# Create a RunPod GPU pod and wait for SSH.  Usage:
#   bash create_pod.sh <name> <gpu_count> [--gpu-type a100|h100|"<full id>"] [--keep-on-timeout]
# e.g.  CONTAINER_DISK_GB=250 bash create_pod.sh my-pod 4 --gpu-type "NVIDIA H200"
# RUNPOD_API_KEY is taken from the environment, else from the repo-root .env.
# The SSH endpoint is saved to .pods/<name>.env next to this script.
#
# Pod-creation constraints:
#   - allowedCudaVersions pins the driver (older drivers refuse the cu13 wheels)
#   - env.PUBLIC_KEY is what actually starts sshd (runpodctl does NOT inject it)
#   - setup.sh installs its own CUDA 13 toolkit, so the stock image works as is
#   - a host whose ports never publish is a crash-looping host: the API keeps
#     saying RUNNING forever, only the console Logs tab shows the error. We
#     fail fast at 20 min (image pull alone can take 15) and TERMINATE the pod
#     so it stops billing; recreate, and after a second failure switch
#     region/pool; after a third, stop — that's platform-side.
set -euo pipefail
NAME="${1:?usage: create_pod.sh <name> <gpu_count> [--gpu-type a100|h100] [--keep-on-timeout]}"
NGPU="${2:?usage: create_pod.sh <name> <gpu_count> [--gpu-type a100|h100] [--keep-on-timeout]}"
shift 2
GPU_TYPE="NVIDIA A100-SXM4-80GB"
KEEP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu-type)
      case "$2" in
        a100) GPU_TYPE="NVIDIA A100-SXM4-80GB" ;;
        h100) GPU_TYPE="NVIDIA H100 80GB HBM3" ;;
        *)    GPU_TYPE="$2" ;;
      esac
      shift 2 ;;
    --keep-on-timeout) KEEP="--keep-on-timeout"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Key from the environment, else the repo-root .env (two levels up).
if [ -z "${RUNPOD_API_KEY:-}" ]; then
  ENV_FILE="$(cd "$(dirname "$0")/../.." && pwd)/.env"
  [ -f "$ENV_FILE" ] && RUNPOD_API_KEY=$(grep -E '^(export )?RUNPOD_API_KEY=' "$ENV_FILE" | tail -1 | sed "s/^export //; s/^RUNPOD_API_KEY=//; s/^['\"]//; s/['\"]\$//")
fi
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not in environment and not found in the repo .env}"
PUBKEY_FILE="${PUBKEY_FILE:-$HOME/.ssh/id_ed25519.pub}"
# Default is the stock RunPod image; IMAGE_NAME overrides it.
IMAGE_NAME="${IMAGE_NAME:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
[ -f "$PUBKEY_FILE" ] || { echo "no ssh pubkey at $PUBKEY_FILE (set PUBKEY_FILE)"; exit 1; }
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DIR/.pods"

BODY=$(PUBKEY="$(cat "$PUBKEY_FILE")" NAME="$NAME" NGPU="$NGPU" GPU_TYPE="$GPU_TYPE" IMAGE_NAME="$IMAGE_NAME" python3 - <<'PY'
import json, os
print(json.dumps({
    "name": os.environ["NAME"],
    "imageName": os.environ["IMAGE_NAME"],
    "gpuTypeIds": [os.environ["GPU_TYPE"]],
    "gpuCount": int(os.environ["NGPU"]),
    "cloudType": "SECURE",
    "containerDiskInGb": int(os.environ.get("CONTAINER_DISK_GB", "150")),
    "volumeInGb": 300,
    "volumeMountPath": "/workspace",
    "ports": ["22/tcp"],
    "allowedCudaVersions": ["13.0"],
    "env": {"PUBLIC_KEY": os.environ["PUBKEY"]},
}))
PY
)

RESP=$(curl -sS -w $'\n%{http_code}' -X POST https://rest.runpod.io/v1/pods \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d "$BODY")
HTTP_CODE=$(printf '%s' "$RESP" | tail -1)
RESP_BODY=$(printf '%s' "$RESP" | sed '$d')
POD_ID=$(printf '%s' "$RESP_BODY" | python3 -c "
import json, sys
try:
    print(json.load(sys.stdin).get('id', ''))
except Exception:
    print('')")
[ -n "$POD_ID" ] || { echo "pod creation failed (HTTP $HTTP_CODE): ${RESP_BODY:-<empty response body>}"; exit 1; }
echo "created pod $POD_ID ($NAME, ${NGPU}x $GPU_TYPE) — waiting for SSH (image pull can take 15 min; billing already running)"

# Poll GraphQL for the published SSH port. NOTE the public port changes on
# every container restart — re-run this query after a restart.
deadline=$(( $(date +%s) + 1200 ))
SSH_IP="" SSH_PORT=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  Q=$(printf '{"query":"query { pod(input: {podId: \\"%s\\"}) { runtime { ports { ip isIpPublic privatePort publicPort } } } }"}' "$POD_ID")
  R=$(curl -sS "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H "Content-Type: application/json" -d "$Q" || true)
  read -r SSH_IP SSH_PORT < <(printf '%s' "$R" | python3 -c "
import json, sys
try:
    ports = json.load(sys.stdin)['data']['pod']['runtime']['ports'] or []
except Exception:
    ports = []
hit = [p for p in ports if p.get('privatePort') == 22 and p.get('isIpPublic')]
print(hit[0]['ip'], hit[0]['publicPort']) if hit else print('', '')
")
  [ -n "$SSH_IP" ] && break
  sleep 20
done

if [ -z "$SSH_IP" ]; then
  echo "SSH never published after 20 min -> crash-looping host (check console Logs tab)."
  if [ "$KEEP" = "--keep-on-timeout" ]; then
    echo "keeping pod $POD_ID as requested — it is still billing."
  else
    curl -sS -X DELETE "https://rest.runpod.io/v1/pods/$POD_ID" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null && echo "terminated $POD_ID."
    echo "recreate once; if it fails again, this pool is sick — change region/pool; third failure = platform-side, stop."
  fi
  exit 1
fi

{
  echo "POD_ID=$POD_ID"
  echo "SSH_IP=$SSH_IP"
  echo "SSH_PORT=$SSH_PORT"
} > "$DIR/.pods/$NAME.env"
echo "ready: ssh -p $SSH_PORT root@$SSH_IP   (saved to .pods/$NAME.env)"
