#!/usr/bin/env bash
# =============================================================================
#  Milestone 5: Production Cloud Cutover — Azure Container Apps + Retrain Job
# =============================================================================
#  Deploys:
#    1. Azure Container Apps Environment (mlops-aca-env)
#    2. Serving Container App (student-performance-api) with external HTTPS
#    3. Scheduled Out-of-Band Retrain Job (student-retrain-worker, every 6h)
#
#  Reads configuration & secrets from .env (git-ignored) and uses your active
#  `az login` subscription (Azure for Students).
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE not found." >&2
  exit 1
fi

# Safely parse .env without bash splitting on ';' inside Azure connection strings
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line#"${line%%[![:space:]]*}"}"
  [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
  key="${line%%=*}"
  val="${line#*=}"
  val="${val%\"}"
  val="${val#\"}"
  export "$key=$val"
done < "$ENV_FILE"

RG="${AZURE_RESOURCE_GROUP:-mlops-platform-rg}"
LOC="${AZURE_LOCATION:-uaenorth}"
ACA_ENV="${ACA_ENVIRONMENT:-mlops-aca-env}"
ACA_JOB_ENV="${ACA_JOB_ENVIRONMENT:-mlops-jobs-env}"
APP_NAME="${ACA_APP_NAME:-student-performance-api}"
JOB_NAME="${ACA_JOB_NAME:-student-retrain-worker}"
IMAGE="${ACA_IMAGE:-ghcr.io/suhaybfr/student-performance-predictor:latest}"
MIN_REPLICAS="${ACA_MIN_REPLICAS:-1}"
MAX_REPLICAS="${ACA_MAX_REPLICAS:-2}"

echo "[Azure] Resource Group : $RG ($LOC)"
echo "[Azure] Container App  : $APP_NAME"
echo "[Azure] Retrain Job    : $JOB_NAME"
echo "[Azure] Image          : $IMAGE"

az extension add --name containerapp --upgrade --only-show-errors >/dev/null 2>&1 || true
az provider register --namespace Microsoft.App --wait >/dev/null
az provider register --namespace Microsoft.OperationalInsights --wait >/dev/null

# 1. Ensure Resource Group exists
az group create --name "$RG" --location "$LOC" -o none

# 2. Ensure Container Apps Environments exist
if ! az containerapp env show --name "$ACA_ENV" --resource-group "$RG" >/dev/null 2>&1; then
  echo "[Azure] Creating Container Apps Environment '$ACA_ENV'..."
  az containerapp env create \
    --name "$ACA_ENV" \
    --resource-group "$RG" \
    --location "$LOC" \
    --environment-mode WorkloadProfiles \
    --no-wait \
    -o none
fi

wait_for_env() {
  local env_name="$1"
  local state=""
  for _ in {1..90}; do
    state="$(az containerapp env show --name "$env_name" --resource-group "$RG" --query properties.provisioningState -o tsv 2>/dev/null || echo "Unknown")"
    if [[ "$state" == "Succeeded" ]]; then
      return 0
    elif [[ "$state" == "Failed" || "$state" == "Canceled" ]]; then
      echo "ERROR: Environment '$env_name' provisioning state: $state" >&2
      return 1
    fi
    sleep 4
  done
}
wait_for_env "$ACA_ENV"

# Express environments don't support Container Apps Jobs; ensure a WorkloadProfiles env exists for jobs
ACA_ENV_MODE="$(az containerapp env show --name "$ACA_ENV" --resource-group "$RG" --query properties.environmentMode -o tsv 2>/dev/null || echo "")"
if [[ "$ACA_ENV_MODE" == "Express" ]]; then
  if ! az containerapp env show --name "$ACA_JOB_ENV" --resource-group "$RG" >/dev/null 2>&1; then
    echo "[Azure] Creating WorkloadProfiles Environment '$ACA_JOB_ENV' for scheduled retrain job..."
    az containerapp env create \
      --name "$ACA_JOB_ENV" \
      --resource-group "$RG" \
      --location "$LOC" \
      --environment-mode WorkloadProfiles \
      --no-wait \
      -o none
  fi
  wait_for_env "$ACA_JOB_ENV"
else
  ACA_JOB_ENV="$ACA_ENV"
fi

# Build secrets & env vars dynamically (Azure Container Apps rejects empty-string secrets)
SECRETS=()
ENV_VARS=(
  "ENABLE_TUTOR=1"
  "GEMINI_MODEL=${GEMINI_MODEL:-gemini-flash-lite-latest}"
  "SUPABASE_URL=${SUPABASE_URL:-}"
  "SUPABASE_STORAGE_BUCKET=${SUPABASE_PDF_BUCKET:-course-pdfs}"
  "AZURE_STORAGE_CONTAINER=${AZURE_STORAGE_CONTAINER:-models}"
  "MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME:-}"
)

add_secret_env() {
  local secret_name="$1"
  local env_name="$2"
  local secret_val="$3"
  if [[ -n "$secret_val" ]]; then
    SECRETS+=("${secret_name}=${secret_val}")
    ENV_VARS+=("${env_name}=secretref:${secret_name}")
  fi
}

add_secret_env "google-api-key" "GOOGLE_API_KEY" "${GOOGLE_API_KEY:-}"
add_secret_env "database-url" "DATABASE_URL" "${DATABASE_URL:-}"
add_secret_env "supabase-service-key" "SUPABASE_SERVICE_ROLE_KEY" "${SUPABASE_SERVICE_ROLE_KEY:-}"
add_secret_env "admin-api-key" "ADMIN_API_KEY" "${ADMIN_API_KEY:-}"
add_secret_env "azure-storage-conn" "AZURE_STORAGE_CONNECTION_STRING" "${AZURE_STORAGE_CONNECTION_STRING:-}"
add_secret_env "mlflow-password" "MLFLOW_TRACKING_PASSWORD" "${MLFLOW_TRACKING_PASSWORD:-}"
add_secret_env "alert-webhook-url" "ALERT_WEBHOOK_URL" "${ALERT_WEBHOOK_URL:-${DISCORD_WEBHOOK_URL:-}}"

wait_for_app() {
  local state=""
  for _ in {1..60}; do
    state="$(az containerapp show --name "$APP_NAME" --resource-group "$RG" --query properties.provisioningState -o tsv 2>/dev/null || echo "Unknown")"
    if [[ "$state" == "Succeeded" ]]; then
      return 0
    elif [[ "$state" == "Failed" || "$state" == "Canceled" ]]; then
      echo "ERROR: Container App '$APP_NAME' provisioning state: $state" >&2
      return 1
    fi
    sleep 4
  done
}

wait_for_job() {
  local state=""
  for _ in {1..60}; do
    state="$(az containerapp job show --name "$JOB_NAME" --resource-group "$RG" --query properties.provisioningState -o tsv 2>/dev/null || echo "Unknown")"
    if [[ "$state" == "Succeeded" ]]; then
      return 0
    elif [[ "$state" == "Failed" || "$state" == "Canceled" ]]; then
      echo "ERROR: Container Apps Job '$JOB_NAME' provisioning state: $state" >&2
      return 1
    fi
    sleep 4
  done
}

# 3. Create or update serving Container App (using --no-wait to avoid CLI socket resets)
if az containerapp show --name "$APP_NAME" --resource-group "$RG" >/dev/null 2>&1; then
  echo "[Azure] Updating existing Container App '$APP_NAME'..."
  wait_for_app
  az containerapp secret set --name "$APP_NAME" --resource-group "$RG" --secrets "${SECRETS[@]}" -o none || wait_for_app
  wait_for_app
  az containerapp update \
    --name "$APP_NAME" \
    --resource-group "$RG" \
    --image "$IMAGE" \
    --min-replicas "$MIN_REPLICAS" \
    --max-replicas "$MAX_REPLICAS" \
    --set-env-vars "${ENV_VARS[@]}" \
    --no-wait \
    -o none
  wait_for_app
else
  echo "[Azure] Creating Container App '$APP_NAME'..."
  az containerapp create \
    --name "$APP_NAME" \
    --resource-group "$RG" \
    --environment "$ACA_ENV" \
    --image "$IMAGE" \
    --target-port 8000 \
    --ingress external \
    --cpu 1.0 \
    --memory 2.0Gi \
    --min-replicas "$MIN_REPLICAS" \
    --max-replicas "$MAX_REPLICAS" \
    --secrets "${SECRETS[@]}" \
    --env-vars "${ENV_VARS[@]}" \
    --no-wait \
    -o none
  wait_for_app
fi

FQDN="$(az containerapp show --name "$APP_NAME" --resource-group "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
APP_URL="https://${FQDN}"
echo "[Azure] Live App URL: $APP_URL"

# 4. Create or update Scheduled Out-of-Band Retrain Job (every 6 hours)
JOB_ENV_VARS=("${ENV_VARS[@]}" "SERVING_API_URL=$APP_URL")
if az containerapp job show --name "$JOB_NAME" --resource-group "$RG" >/dev/null 2>&1; then
  echo "[Azure] Updating scheduled Container Apps Job '$JOB_NAME'..."
  wait_for_job
  az containerapp job secret set --name "$JOB_NAME" --resource-group "$RG" --secrets "${SECRETS[@]}" -o none || wait_for_job
  wait_for_job
  az containerapp job update \
    --name "$JOB_NAME" \
    --resource-group "$RG" \
    --image "$IMAGE" \
    --set-env-vars "${JOB_ENV_VARS[@]}" \
    --no-wait \
    -o none
  wait_for_job
else
  echo "[Azure] Creating scheduled Container Apps Job '$JOB_NAME' (every 6h)..."
  az containerapp job create \
    --name "$JOB_NAME" \
    --resource-group "$RG" \
    --environment "$ACA_JOB_ENV" \
    --trigger-type Schedule \
    --cron-expression "0 */6 * * *" \
    --replica-timeout 900 \
    --replica-retry-limit 1 \
    --parallelism 1 \
    --replica-completion-count 1 \
    --image "$IMAGE" \
    --cpu 1.0 \
    --memory 2.0Gi \
    --command "python" \
    --args "src/worker_retrain.py" \
    --secrets "${SECRETS[@]}" \
    --env-vars "${JOB_ENV_VARS[@]}" \
    --no-wait \
    -o none
  wait_for_job
fi

# Ensure worker runs in --shadow mode via ARM PATCH (preserves redacted secrets & avoids az argparse)
JOB_ID="$(az containerapp job show --name "$JOB_NAME" --resource-group "$RG" --query id -o tsv)"
CONTAINER_JSON="$(az containerapp job show --name "$JOB_NAME" --resource-group "$RG" --query "properties.template.containers[0]" -o json | python3 -c 'import sys,json; c=json.load(sys.stdin); c.pop("imageType",None); c["args"]=["src/worker_retrain.py","--shadow"]; print(json.dumps({"properties":{"template":{"containers":[c]}}}))')"
az rest --method PATCH --uri "https://management.azure.com${JOB_ID}?api-version=2024-03-01" --body "$CONTAINER_JSON" -o none || wait_for_job
wait_for_job

echo ""
echo "[Azure] Smoke-checking ${APP_URL}/health ..."
curl -fsS "${APP_URL}/health" && echo ""
echo ""
echo "============================================================================="
echo " Deployment Complete!"
echo "   Web UI    : ${APP_URL}/tutor/"
echo "   API Docs  : ${APP_URL}/docs"
echo "   Metrics   : ${APP_URL}/metrics  (requires Authorization: Bearer <ADMIN_API_KEY>)"
echo "============================================================================="
