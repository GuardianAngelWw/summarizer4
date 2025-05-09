# Telegram Bot on Google Cloud Run (Always Free Tier)

This repository contains a Telegram bot designed to run on Google Cloud Run using the "Always Free" tier resources.

## Features

- Runs on Google Cloud Run's "Always Free" tier
- Data persistence with Google Firestore
- Webhook-based setup for optimal performance
- Automatic scaling (within free tier limits)
- Continuous deployment via Cloud Build
- Secret management with Google Secret Manager

## Prerequisites

- Google Cloud account (credit card required for identity verification, but you won't be charged)
- Telegram Bot token (obtain from [@BotFather](https://t.me/BotFather))
- Groq API key for AI responses
- Google Cloud SDK installed locally (for deployment)

## Deployment Options

### Option 1: Cloud Build Automatic Deployment (NEW)

This is the simplest way to deploy your bot with continuous integration:

1. Create a new Google Cloud project
2. Set up required secrets in Secret Manager
3. Connect your repository to Cloud Build
4. The bot will be automatically deployed and configured

**Detailed steps:**

1. **Create a new project and set it as active:**
   ```bash
   gcloud projects create your-project-name
   gcloud config set project your-project-name
   ```

2. **Create and set up the required secrets:**
   ```bash
   # Enable Secret Manager API
   gcloud services enable secretmanager.googleapis.com
   
   # Create secrets
   gcloud secrets create telegram-bot-token --replication-policy=automatic
   gcloud secrets create groq-api-key --replication-policy=automatic
   gcloud secrets create admin-user-ids --replication-policy=automatic
   
   # Add values to secrets
   echo -n "YOUR_BOT_TOKEN" | gcloud secrets versions add telegram-bot-token --data-file=-
   echo -n "YOUR_GROQ_API_KEY" | gcloud secrets versions add groq-api-key --data-file=-
   echo -n "YOUR_ADMIN_ID1,YOUR_ADMIN_ID2" | gcloud secrets versions add admin-user-ids --data-file=-
   ```

3. **Connect repository to Cloud Build:**
   - Go to Google Cloud Console > Cloud Build > Triggers
   - Click "Connect Repository" and select your repository
   - Create a new trigger:
     - Name: "Deploy Telegram Bot"
     - Event: Push to a branch
     - Source: Your repository branch (e.g., main)
     - Configuration: Repository (Cloud Build configuration file)
     - Location: `/cloudbuild.yaml`
   - Click "Create"

4. **Run the trigger manually the first time:**
   - In the triggers list, click "Run" on your new trigger
   - This initial build will:
     - Enable all required services
     - Set up Firestore database
     - Create and configure service accounts
     - Build and deploy your bot
     - Set up the Telegram webhook

After this setup, any changes pushed to your repository will automatically trigger a rebuild and redeployment of your bot.

### Option 2: One-Click Deployment Script

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Make the deployment script executable:
   ```bash
   chmod +x deployment-commands.sh
   ```

3. Set environment variables:
   ```bash
   export BOT_TOKEN="your_telegram_bot_token"
   export GROQ_API_KEY="your_groq_api_key"
   export ADMIN_IDS="12345678,87654321"
   export PROJECT_NAME="your-project-name"  # Optional
   ```

4. Run the deployment script:
   ```bash
   ./deployment-commands.sh
   ```

### Option 3: Manual Deployment

For a step-by-step manual deployment, follow these instructions:

1. **Setup Google Cloud Project:**
   ```bash
   # Create a new project
   gcloud projects create your-project-name --name="Telegram Bot"
   gcloud config set project your-project-name
   
   # Enable required services
   gcloud services enable cloudbuild.googleapis.com run.googleapis.com secretmanager.googleapis.com firestore.googleapis.com
   ```

2. **Store Secrets:**
   ```bash
   # Create secrets
   echo -n "your_bot_token" | gcloud secrets create telegram-bot-token --data-file=-
   echo -n "your_groq_api_key" | gcloud secrets create groq-api-key --data-file=-
   echo -n "12345678,87654321" | gcloud secrets create admin-user-ids --data-file=-
   ```

3. **Create Firestore Database:**
   ```bash
   gcloud firestore databases create --region=us-central1
   ```

4. **Set Up Service Account:**
   ```bash
   # Create service account
   gcloud iam service-accounts create telegram-bot-sa
   
   # Grant permissions
   gcloud projects add-iam-policy-binding your-project-name \
     --member="serviceAccount:telegram-bot-sa@your-project-name.iam.gserviceaccount.com" \
     --role="roles/datastore.user"
     
   gcloud projects add-iam-policy-binding your-project-name \
     --member="serviceAccount:telegram-bot-sa@your-project-name.iam.gserviceaccount.com" \
     --role="roles/secretmanager.secretAccessor"
   ```

5. **Build and Deploy:**
   ```bash
   # Build using Cloud Build
   gcloud builds submit --tag gcr.io/your-project-name/telegram-bot:v1
   
   # Deploy to Cloud Run
   gcloud run deploy telegram-bot \
     --image gcr.io/your-project-name/telegram-bot:v1 \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 256Mi \
     --cpu 1 \
     --max-instances 1 \
     --service-account telegram-bot-sa@your-project-name.iam.gserviceaccount.com \
     --set-secrets="BOT_TOKEN=telegram-bot-token:latest,GROQ_API_KEY=groq-api-key:latest,ADMIN_USER_IDS=admin-user-ids:latest"
   ```

6. **Set Webhook URL:**
   ```bash
   # Get the service URL
   SERVICE_URL=$(gcloud run services describe telegram-bot --platform managed --region us-central1 --format 'value(status.url)')
   
   # Update the service with the webhook URL
   gcloud run services update telegram-bot \
     --platform managed \
     --region us-central1 \
     --set-env-vars="WEBHOOK_URL=$SERVICE_URL"
   
   # Set the webhook in Telegram
   curl -X POST "https://api.telegram.org/bot$BOT_TOKEN/setWebhook?url=$SERVICE_URL/$BOT_TOKEN"
   ```

## Free Tier Limitations

- **Cloud Run**: 2 million requests per month, 360,000 GB-seconds of compute/month
- **Firestore**: 1 GB storage, 50,000 reads/day, 20,000 writes/day, 20,000 deletes/day
- **Cloud Build**: 120 build-minutes per day
- **Secret Manager**: 6 secret versions

To stay within free tier limits, this deployment uses:
- 1 Cloud Run instance with 256MB memory
- Minimal Firestore operations with fallback to local storage
- Infrequent deployments to minimize Cloud Build usage

## Monitoring and Maintenance

- **Check logs:**
  ```bash
  gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=telegram-bot" --limit 10
  ```

- **Check service status:**
  ```bash
  gcloud run services describe telegram-bot --region us-central1
  ```

- **Create budget alert (recommended):**
  ```bash
  gcloud billing budgets create --billing-account=YOUR_BILLING_ACCOUNT_ID \
    --display-name="Telegram Bot Budget" \
    --budget-amount=0USD \
    --threshold-rule=percent=80 \
    --threshold-rule=percent=100
  ```

## Project Structure

- `webhook_handler.py`: Main entry point for Cloud Run
- `ollama_telegram_bot.py`: Core bot functionality
- `firestore_storage.py`: Data persistence with Firestore
- `Dockerfile`: Container definition
- `cloudbuild.yaml`: CI/CD configuration
- `app.yaml`: App Engine configuration (alternative deployment)
- `deployment-commands.sh`: One-click deployment script 