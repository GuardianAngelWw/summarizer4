#!/bin/bash
# Deployment script for Telegram Bot to Google Cloud Run (Always Free Tier)
# This script automates the deployment process to keep within free tier limits

# Exit on error
set -e

echo "========== Telegram Bot Deployment (Google Cloud Run Free Tier) =========="
echo "This script will deploy your Telegram bot to Google Cloud Run"
echo "Prerequisites:"
echo " - Google Cloud SDK installed"
echo " - gcloud initialized with your account"
echo " - Docker installed (only if building locally)"
echo "======================================================================"

# Set environment variables
BOT_TOKEN=${BOT_TOKEN:-"YOUR_BOT_TOKEN"}
GROQ_API_KEY=${GROQ_API_KEY:-"YOUR_GROQ_API_KEY"}
ADMIN_IDS=${ADMIN_IDS:-"COMMA_SEPARATED_ADMIN_IDS"}
PROJECT_NAME=${PROJECT_NAME:-"telegram-bot-$(date +%s)"}
REGION=${REGION:-"us-central1"}

echo "Checking if gcloud is installed..."
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud is not installed. Please install the Google Cloud SDK."
    exit 1
fi

# Create new project if needed
echo "Creating new Google Cloud project: $PROJECT_NAME"
gcloud projects create $PROJECT_NAME --name="Telegram Bot"

# Set the project as active
echo "Setting project as active..."
gcloud config set project $PROJECT_NAME

# Enable required services
echo "Enabling required Google Cloud services..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable firestore.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Create Firestore database (in Native mode)
echo "Creating Firestore database in Native mode..."
gcloud firestore databases create --region=$REGION --quiet

# Store secrets in Secret Manager
echo "Storing secrets in Secret Manager..."
echo -n "$BOT_TOKEN" | gcloud secrets create telegram-bot-token --data-file=-
echo -n "$GROQ_API_KEY" | gcloud secrets create groq-api-key --data-file=-
echo -n "$ADMIN_IDS" | gcloud secrets create admin-user-ids --data-file=-

# Build the container
echo "Building container using Cloud Build..."
gcloud builds submit --tag gcr.io/$PROJECT_NAME/telegram-bot:v1

# Get service account
SERVICE_ACCOUNT=$(gcloud iam service-accounts list --filter="email ~ $PROJECT_NAME" --format="value(email)" | head -n 1)

if [ -z "$SERVICE_ACCOUNT" ]; then
    echo "Creating service account..."
    SERVICE_ACCOUNT="$PROJECT_NAME@$PROJECT_NAME.iam.gserviceaccount.com"
    gcloud iam service-accounts create $PROJECT_NAME
fi

# Grant permissions to the service account
echo "Granting necessary permissions to service account..."
gcloud projects add-iam-policy-binding $PROJECT_NAME \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/datastore.user"

gcloud projects add-iam-policy-binding $PROJECT_NAME \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

# Deploy to Cloud Run with minimum resources (free tier)
echo "Deploying to Cloud Run..."
gcloud run deploy telegram-bot \
    --image gcr.io/$PROJECT_NAME/telegram-bot:v1 \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 256Mi \
    --cpu 1 \
    --max-instances 1 \
    --service-account $SERVICE_ACCOUNT \
    --set-secrets="BOT_TOKEN=telegram-bot-token:latest,GROQ_API_KEY=groq-api-key:latest,ADMIN_USER_IDS=admin-user-ids:latest"

# Get the service URL
SERVICE_URL=$(gcloud run services describe telegram-bot --platform managed --region $REGION --format 'value(status.url)')

# Update the service with the webhook URL
echo "Updating service with webhook URL..."
gcloud run services update telegram-bot \
    --platform managed \
    --region $REGION \
    --set-env-vars="WEBHOOK_URL=$SERVICE_URL"

# Set the webhook in Telegram
echo "Setting webhook URL in Telegram..."
curl -X POST "https://api.telegram.org/bot$BOT_TOKEN/setWebhook?url=$SERVICE_URL/$BOT_TOKEN"

# Done!
echo "======================================================================"
echo "Deployment complete! Your bot should now be running on Google Cloud Run."
echo "Service URL: $SERVICE_URL"
echo ""
echo "To check logs, run:"
echo "gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=telegram-bot\" --limit 10"
echo ""
echo "To check status, run:"
echo "gcloud run services describe telegram-bot --region $REGION"
echo ""
echo "Remember: Free tier includes 2 million requests/month, but your service"
echo "is limited to 1 instance with 256MB memory to stay within free limits."
echo "======================================================================" 