#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-default}"

kubectl create configmap cpt-a3mega-70b-code \
  --namespace "${NAMESPACE}" \
  --from-file=train_cpt.py="${SCRIPT_DIR}/train_cpt.py" \
  --from-file=fsdp_config.yaml="${SCRIPT_DIR}/fsdp_config.yaml" \
  --dry-run=client \
  -o yaml | kubectl apply --namespace "${NAMESPACE}" -f -

echo "Updated cpt-a3mega-70b-code in namespace ${NAMESPACE}."
