#!/usr/bin/env bash
# Keep an SSH tunnel to a serving pod alive for the length of an eval.
#
#   IP=<pod-ip> PORT=<ssh-port> LOCAL=8001 bash tunnel_keeper.sh &
#
# Why this exists rather than a plain `ssh -f -N -L`: a tunnel that dies
# mid-run does NOT surface as an error. Inspect swallows the dropped
# connections as retries with escalating backoff, so a dead tunnel looks
# exactly like a slow model. ServerAliveInterval alone does not prevent it.
#
# So this does not trust the ssh process to still be working. It probes the
# forwarded port every 20s with a real HTTP request to /v1/models, and rebuilds
# the tunnel whenever the probe fails. Watch the log, or watch GPU utilization
# on the pod — never "is the ssh process alive", which is the thing that lies.
set -uo pipefail
IP="${IP:?set IP to the pod address}"
PORT="${PORT:?set PORT to the pod ssh port}"
LOCAL="${LOCAL:?set LOCAL to the local port to forward}"

start_tunnel() {
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes -N -L "${LOCAL}:127.0.0.1:8000" \
      -p "$PORT" "root@$IP" &
  TUNNEL_PID=$!
}

cleanup() { kill "${TUNNEL_PID:-0}" 2>/dev/null; }
trap cleanup EXIT

start_tunnel
echo "$(date +%H:%M:%S) tunnel up on :$LOCAL (pid $TUNNEL_PID)"
FAILS=0
while true; do
  sleep 20
  if curl -sf -m 10 "http://127.0.0.1:${LOCAL}/v1/models" >/dev/null 2>&1; then
    FAILS=0
    continue
  fi
  # One failed probe can just be a busy server mid-generation; two in a row
  # (40s) means the forward is gone. Rebuild rather than wait for ssh to notice.
  FAILS=$((FAILS + 1))
  [ "$FAILS" -lt 2 ] && continue
  echo "$(date +%H:%M:%S) probe failed ${FAILS}x — rebuilding tunnel"
  kill "$TUNNEL_PID" 2>/dev/null
  sleep 2
  start_tunnel
  FAILS=0
done
