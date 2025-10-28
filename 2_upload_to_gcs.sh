#!/usr/bin/env bash
# ========================================================
# Upload local scikit-learn model artifacts to a GCS bucket
# for Vertex AI deployment
# ========================================================

# --- CONFIGURATION ---
PROJECT_ID="fifth-sprite-475713-q9"         # 👈 Twój projekt
REGION="us-central1"                         # np. europe-west4 / us-central1
BUCKET_NAME="fifth-sprite-475713-q9-bucket0" # bucket musi istnieć
MODEL_NAME="sklogit_pd_1m"                   # nazwa folderu/modelu
LOCAL_ARTIFACT_DIR="model_artifacts"         # zawiera model.joblib + feature_list.json

set -euo pipefail

echo "🔧 Setting project..."
gcloud config set project "${PROJECT_ID}"

echo "🧪 Checking required file: ${LOCAL_ARTIFACT_DIR}/model.joblib"
test -f "${LOCAL_ARTIFACT_DIR}/model.joblib" || {
  echo "❌ Brak ${LOCAL_ARTIFACT_DIR}/model.joblib. Uruchom najpierw: python train_logit.py"
  exit 1
}

echo "🪣 Checking bucket gs://${BUCKET_NAME}..."
if ! gsutil ls -b "gs://${BUCKET_NAME}" >/dev/null 2>&1; then
  echo "❌ Bucket gs://${BUCKET_NAME} nie istnieje. Utwórz go:"
  echo "   gsutil mb -l ${REGION} gs://${BUCKET_NAME}"
  exit 1
fi

DESTINATION="gs://${BUCKET_NAME}/${MODEL_NAME}/"
echo "⬆️  Uploading artifacts to ${DESTINATION}"
gsutil -m rsync -r "${LOCAL_ARTIFACT_DIR}" "${DESTINATION}"

echo "✅ Uploaded files:"
gsutil ls -r "${DESTINATION}"

echo "🚀 All done! Use this ARTIFACT_URI in deploy:"
echo "    ARTIFACT_URI=${DESTINATION}"
