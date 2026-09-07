#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-default}"

kubectl create configmap cpt-a3mega-code \
  --namespace "${NAMESPACE}" \
  --from-file=train_cpt.py="${SCRIPT_DIR}/train_cpt.py" \
  --dry-run=client \
  -o yaml | kubectl apply --namespace "${NAMESPACE}" -f -

echo "Updated ConfigMap cpt-a3mega-code in namespace ${NAMESPACE}."
echo "Restart the JobSet to make a running training process load the new code."
