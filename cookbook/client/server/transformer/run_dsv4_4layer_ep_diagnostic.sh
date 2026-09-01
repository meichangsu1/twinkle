#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-}"
MODE="${2:-}"

if [[ "$ROLE" != "head" && "$ROLE" != "worker" ]]; then
    echo "Usage: $0 {head|worker} {no_ep|ep_loop|ep_gmm}" >&2
    exit 2
fi
if [[ "$MODE" != "no_ep" && "$MODE" != "ep_loop" && "$MODE" != "ep_gmm" ]]; then
    echo "Usage: $0 {head|worker} {no_ep|ep_loop|ep_gmm}" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case "$MODE" in
    no_ep)
        export TWINKLE_SERVER_CONFIG_PATH="$SCRIPT_DIR/server_config_dsv4_4layer_diag_no_ep.yaml"
        export TWINKLE_EP_FORCE_LOOP=0
        ;;
    ep_loop)
        export TWINKLE_SERVER_CONFIG_PATH="$SCRIPT_DIR/server_config_dsv4_4layer_diag_ep.yaml"
        export TWINKLE_EP_FORCE_LOOP=1
        ;;
    ep_gmm)
        export TWINKLE_SERVER_CONFIG_PATH="$SCRIPT_DIR/server_config_dsv4_4layer_diag_ep.yaml"
        export TWINKLE_EP_FORCE_LOOP=0
        ;;
esac

export TWINKLE_EP_DIAGNOSTICS="${TWINKLE_EP_DIAGNOSTICS:-1}"
export SKIP_DATASET_CHECK=1

echo "Starting DeepSeek-V4 four-layer diagnostic: role=$ROLE mode=$MODE"
exec "$SCRIPT_DIR/run_dsv4_0731_npu_2node_2npu.sh" "$ROLE"
