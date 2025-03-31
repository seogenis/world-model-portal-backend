# Deploying on Brev

This document provides instructions for deploying the Cosmos Text2World Prompt Tuning API on a Brev server instance.

## Prerequisites

- A Brev account and server instance
- SSH access to your Brev instance
- NVIDIA API key for Cosmos
- OpenAI API key

## Deployment Steps

### 1. Clone the Repository

SSH into your Brev instance and clone the repository:

```bash
git clone https://github.com/yourusername/cosmos-prompt-tuner.git
cd cosmos-prompt-tuner
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key
NVIDIA_API_KEY=your_nvidia_api_key
EOF
```

### 3. Setup Options

#### Option 1: Manual Setup

```bash
# Run the setup script
chmod +x setup.sh
./setup.sh

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### Option 2: Docker Setup

```bash
# Build and start with Docker Compose
docker-compose up -d
```

### 4. Configure Persistent Service (Optional)

To ensure the service continues running even after you disconnect, set up a systemd service:

```bash
# Create a systemd service file
sudo tee /etc/systemd/system/cosmos-api.service > /dev/null << EOF
[Unit]
Description=Cosmos Text2World Prompt Tuning API
After=network.target

[Service]
User=$(whoami)
WorkingDirectory=$(pwd)
ExecStart=$(which uvicorn) app.main:app --host 0.0.0.0 --port 8000
Restart=always
StandardOutput=journal
StandardError=journal
Environment="PYTHONPATH=$(pwd)"
EnvironmentFile=$(pwd)/.env

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl enable cosmos-api
sudo systemctl start cosmos-api

# Check status
sudo systemctl status cosmos-api
```

### 5. Access Your API

Your API will now be available at the public IP or domain of your Brev instance:

```
http://your-brev-instance-ip:8000/
```

You can access the API documentation at:

```
http://your-brev-instance-ip:8000/docs
```

### 6. Using a Domain Name (Optional)

If you want to use a custom domain with your API:

1. Set up DNS records to point your domain to your Brev instance IP
2. Install and configure Nginx as a reverse proxy

```bash
# Install Nginx
sudo apt update
sudo apt install nginx

# Configure Nginx
sudo tee /etc/nginx/sites-available/cosmos-api.conf > /dev/null << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable site and restart Nginx
sudo ln -s /etc/nginx/sites-available/cosmos-api.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 7. Updating the Frontend Examples

Update the frontend examples to use your Brev instance URL:

```bash
./update_frontend_urls.sh http://your-brev-instance-ip:8000
```

## Troubleshooting

### Server Not Starting

Check the service logs:

```bash
sudo journalctl -u cosmos-api
```

### Permission Issues

Make sure the user has appropriate permissions:

```bash
chmod -R 755 .
chmod +x *.sh
```

### API Key Issues

Check that your API keys are correctly set in the `.env` file:

```bash
cat .env
```