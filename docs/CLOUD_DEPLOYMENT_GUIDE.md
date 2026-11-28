# Milestone 5 — Production Cloud Cutover Guide

This guide covers deploying the full **AI-Based Digital Literacy Assistant** to **Azure Container Apps** (using Azure for Students credits), wiring **GitHub Actions CD**, connecting **Grafana Cloud**, enabling **Discord / Email alert paging**, and attaching a **Cloudflare custom domain**.

---

## 1. Deploy Serving API + Scheduled Retrain Job to Azure Container Apps

Make sure you are signed into Azure CLI (`az login`) and your `.env` contains your Supabase, Gemini, Azure Blob, DAGsHub MLflow, and Admin API keys, then run:

```bash
chmod +x infra/deploy_azure.sh
./infra/deploy_azure.sh
```

This idempotently creates/updates:
1. **`mlops-aca-env`** — Azure Container Apps Environment in `uaenorth` (`mlops-platform-rg`)
2. **`student-performance-api`** — Public HTTPS FastAPI + Tutor + Online Feature Store app (`ghcr.io/suhaybfr/student-performance-predictor:latest`, port `8000`)
3. **`student-retrain-worker`** — Scheduled Azure Container Apps Job (`0 */6 * * *`, every 6 hours) running `python -m src.worker_retrain --from-live --shadow --api-url https://<fqdn>`

---

## 2. Enable Automated CD in GitHub Actions (`.github/workflows/ci.yml`)

Create an Azure Service Principal scoped to `mlops-platform-rg`:

```bash
az ad sp create-for-rbac \
  --name "mlops-github-cd" \
  --role Contributor \
  --scopes "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/mlops-platform-rg" \
  --sdk-auth
```

Add the JSON output as a GitHub Repository Secret named **`AZURE_CREDENTIALS`** (Settings → Secrets and variables → Actions).  
On every push to `main`, the 6th CI job (`deploy-azure`) will automatically roll out the newly built GHCR image to both `student-performance-api` and `student-retrain-worker` and verify `/health`.

---

## 3. Connect Grafana Cloud + Discord / Email Alert Paging

### A. Live Discord Webhook Paging (Direct from App & Retrain Worker)
1. In Discord: **Server Settings → Integrations → Webhooks → New Webhook** → copy the URL.
2. Add `ALERT_WEBHOOK_URL=https://discord.com/api/webhooks/...` to `.env` (and re-run `./infra/deploy_azure.sh`).
3. Test live paging immediately:
   ```bash
   curl -s -X POST "https://<your-aca-fqdn>/alerts/test" \
     -H "X-API-Key: <ADMIN_API_KEY>" | python3 -m json.tool
   ```
   Drift detections (`dataset_drifted` / `model_drift_detected`), quality-gate failures, and model promotions now page Discord automatically!

### B. Grafana Cloud Dashboard & Prometheus Scraping
1. Sign up for the **Grafana Cloud Free Tier** (`grafana.com`).
2. Import **`monitoring/grafana/dashboards/mlops_dashboard.json`** into Grafana Cloud (**Dashboards → New → Import**).
3. Scrape `https://<your-aca-fqdn>/metrics` using **Grafana Cloud Metrics Endpoint** or Grafana Alloy with **`monitoring/grafana-cloud/alloy-config.river`** (passing `Authorization: Bearer <ADMIN_API_KEY>`).
4. In Grafana Cloud **Alerting → Contact points**, add your **Discord Webhook** and/or **Email** and import the 9 rules from **`monitoring/prometheus/alert_rules.yml`**.

---

## 4. Attach a Custom Domain via Cloudflare (GitHub Student Pack)

1. Claim a free domain (e.g. `.me` / `.tech` / `.live`) via **GitHub Student Developer Pack** and add it to **Cloudflare** (Free DNS plan).
2. In Azure Portal → **`student-performance-api` → Custom domains → Add custom domain**:
   - Enter your hostname (e.g. `mlops.yourdomain.me`).
   - Azure shows two DNS records to add in Cloudflare:
     - `CNAME` `mlops` → `<your-aca-fqdn>` (set Cloudflare Proxy to **DNS only / grey cloud** during initial TLS certificate issuance, then optionally turn on orange cloud).
     - `TXT` `asuid.mlops` → `<azure-verification-id>`.
3. Click **Validate → Add** and select **Managed certificate (Free)** for automatic HTTPS TLS!
