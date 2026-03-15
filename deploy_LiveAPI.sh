#!/bin/bash
# deploy.sh — One command to deploy Running Coach to GCP
# Usage: ./deploy_LiveAPI.sh

set -e  # Exit on any error

PROJECT_ID="running-assistant-485215"
#REGION="europe-west1"
REGION="us-central1"
#SERVICE_NAME="running-coach-google-cloud-tts"
#SERVICE_NAME="running-coach-gemini-tts"
SERVICE_NAME="running-coach-live-api"
DB_INSTANCE="running-coach-db"
DB_NAME="running_coach"
DB_USER="coach_user"
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Deploying Running Coach to GCP..."
echo "Project: $PROJECT_ID | Region: $REGION"
echo ""

# ── Step 1: Create Cloud SQL instance ─────────────────────
echo "📦 Step 1: Setting up Cloud SQL PostgreSQL..."
if gcloud sql instances describe $DB_INSTANCE --project=$PROJECT_ID &>/dev/null; then
    echo "✅ Database instance already exists, skipping creation."
else
    gcloud sql instances create $DB_INSTANCE \
        --database-version=POSTGRES_15 \
        --tier=db-f1-micro \
        --region=$REGION \
        --project=$PROJECT_ID \
        --storage-size=10GB \
        --storage-type=SSD

    echo "✅ Cloud SQL instance created."
fi

# Create database
gcloud sql databases create $DB_NAME \
    --instance=$DB_INSTANCE \
    --project=$PROJECT_ID 2>/dev/null || echo "✅ Database already exists."

# Set password for user
DB_PASSWORD=$(openssl rand -base64 20)
gcloud sql users create $DB_USER \
    --instance=$DB_INSTANCE \
    --password=$DB_PASSWORD \
    --project=$PROJECT_ID 2>/dev/null || echo "✅ User already exists."

# Get connection name
DB_CONNECTION=$(gcloud sql instances describe $DB_INSTANCE \
    --project=$PROJECT_ID \
    --format="value(connectionName)")

echo "✅ Database ready. Connection: $DB_CONNECTION"
echo ""

# ── Step 2: Store secrets in Secret Manager ───────────────
echo "🔐 Step 2: Storing secrets..."

# Read from local .env file
if [ ! -f .env ]; then
    echo "❌ .env file not found! Please run from the running_coach_server directory."
    exit 1
fi

GEMINI_KEY=$(grep GEMINI_API_KEY .env | cut -d '=' -f2 | tr -d '"' | tr -d ' ')

# Store Gemini key (create or update)
if [ -n "$GEMINI_KEY" ]; then
    echo -n "$GEMINI_KEY" | gcloud secrets create gemini-api-key \
        --data-file=- --project=$PROJECT_ID 2>/dev/null || \
    echo -n "$GEMINI_KEY" | gcloud secrets versions add gemini-api-key \
        --data-file=- --project=$PROJECT_ID
    echo "✅ Gemini API key stored."
else
    echo "⚠️  GEMINI_API_KEY not in .env — assuming already in Secret Manager."
fi

# Build DATABASE_URL for Cloud SQL
DATABASE_URL="postgresql://$DB_USER:$DB_PASSWORD@/$DB_NAME?host=/cloudsql/$DB_CONNECTION"
echo -n "$DATABASE_URL" | gcloud secrets create database-url \
    --data-file=- --project=$PROJECT_ID 2>/dev/null || \
echo -n "$DATABASE_URL" | gcloud secrets versions add database-url \
    --data-file=- --project=$PROJECT_ID

echo "✅ Secrets stored."
echo ""

# ── Step 3: Build and push Docker image ───────────────────
echo "🐳 Step 3: Building Docker image..."
gcloud auth configure-docker --quiet
docker build --platform linux/amd64 -t $IMAGE .
docker push $IMAGE
echo "✅ Image pushed: $IMAGE"
echo ""

# ── Step 4: Deploy to Cloud Run ───────────────────────────
echo "☁️  Step 4: Deploying to Cloud Run..."

# Get service account email
SA_EMAIL=$(gcloud iam service-accounts list \
    --filter="displayName:Compute Engine default service account" \
    --format="value(email)" \
    --project=$PROJECT_ID)

# Grant secret access
gcloud secrets add-iam-policy-binding gemini-api-key \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/secretmanager.secretAccessor" \
    --project=$PROJECT_ID 2>/dev/null || true

gcloud secrets add-iam-policy-binding database-url \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/secretmanager.secretAccessor" \
    --project=$PROJECT_ID 2>/dev/null || true

gcloud run deploy $SERVICE_NAME \
    --image=$IMAGE \
    --platform=managed \
    --region=$REGION \
    --allow-unauthenticated \
    --add-cloudsql-instances=$DB_CONNECTION \
    --set-secrets="DATABASE_URL=database-url:latest,GEMINI_API_KEY=gemini-api-key:latest" \
    --set-env-vars="GCP_PROJECT=running-assistant-485215,GCP_REGION=us-central1" \
    --memory=1Gi \
    --cpu=1 \
    --min-instances=1 \
    --max-instances=10 \
    --port=8080 \
    --timeout=3600 \
    --session-affinity \
    --project=$PROJECT_ID

echo ""
echo "✅ Deployment complete!"
echo ""

# ── Get the URL ───────────────────────────────────────────
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --platform=managed \
    --region=$REGION \
    --project=$PROJECT_ID \
    --format="value(status.url)" 2>/dev/null)

# Fallback: construct URL from project number if describe fails
if [ -z "$SERVICE_URL" ]; then
    PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
    SERVICE_URL="https://$SERVICE_NAME-$PROJECT_NUMBER.$REGION.run.app"
fi

echo "🎉 Your server is live at:"
echo "   $SERVICE_URL"
echo ""
echo "📱 Update ServerConfig.swift:"
echo "   static let baseURL = \"$SERVICE_URL\""
echo "   static let wsURL   = \"${SERVICE_URL/https/wss}\""
echo ""
echo "🏃 You can now run without your Mac!"

# ── Helper: store Gemini key (run once) ───────────────────
# echo -n "YOUR_GEMINI_KEY" | gcloud secrets create gemini-api-key --data-file=- --project=$PROJECT_ID
# Or update existing:
# echo -n "YOUR_GEMINI_KEY" | gcloud secrets versions add gemini-api-key --data-file=- --project=$PROJECT_ID