# EC2 Deployment Guide — Vitality Leisure Park

## What you need
- AWS account
- Your project files (this folder + model files from GitHub)
- Your Cohere API key

---

## Step 1 — Launch EC2 Instance

1. Go to AWS Console → EC2 → Launch Instance
2. Settings:
   - **Name**: vitality-leisure
   - **AMI**: Ubuntu Server 24.04 LTS (free tier eligible)
   - **Instance type**: t2.micro (free tier) or t2.small (better performance)
   - **Key pair**: Create new → download .pem file → KEEP THIS SAFE
   - **Security group**: Add these inbound rules:
     - SSH (port 22) from My IP
     - HTTP (port 80) from Anywhere
     - HTTPS (port 443) from Anywhere
     - Custom TCP port 5000 from Anywhere (for testing)
3. Click Launch

---

## Step 2 — Connect to your instance

On your Mac, open Terminal:

```bash
# Make key file secure
chmod 400 ~/Downloads/your-key.pem

# Connect (replace with your EC2 public IP from AWS console)
ssh -i ~/Downloads/your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

---

## Step 3 — Install dependencies on EC2

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python, pip, nginx
sudo apt install -y python3 python3-pip python3-venv nginx git

# Create app directory
mkdir -p /home/ubuntu/vitality
cd /home/ubuntu/vitality
```

---

## Step 4 — Upload your files

On your **local Mac** (new terminal window):

```bash
# Upload everything from your project folder
scp -i ~/Downloads/your-key.pem -r \
  /Users/verakannenwischer/Downloads/cloudassignment12/* \
  ubuntu@YOUR_EC2_PUBLIC_IP:/home/ubuntu/vitality/

# Also upload the web app folder
scp -i ~/Downloads/your-key.pem -r \
  /path/to/vitality-web/* \
  ubuntu@YOUR_EC2_PUBLIC_IP:/home/ubuntu/vitality/
```

Your EC2 should now have:
```
/home/ubuntu/vitality/
├── app.py                  (Flask app)
├── model.joblib
├── model_meta.json
├── monthly_avg.json
├── weekday_avg.json
├── yearmonth_avg.csv
├── data.xlsx
├── embeddings.json
├── Menu_RAG.pdf
├── Fitness_RAG.pdf
├── requirements.txt
├── static/
│   ├── css/main.css
│   ├── js/main.js
│   └── images/forest_bg.png
└── templates/
    ├── base.html
    ├── index.html
    ├── manager.html
    ├── visitor.html
    └── wellness.html
```

---

## Step 5 — Set up Python environment

Back in the **EC2 terminal**:

```bash
cd /home/ubuntu/vitality

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your Cohere API key (permanent)
echo 'export COHERE_API_KEY="your-actual-key-here"' >> ~/.bashrc
source ~/.bashrc

# Test the app runs
python3 app.py
# Should see: Running on http://0.0.0.0:5000
# Press Ctrl+C to stop
```

---

## Step 6 — Set up Gunicorn (production server)

```bash
# Create systemd service so app auto-starts
sudo nano /etc/systemd/system/vitality.service
```

Paste this (replace YOUR_EC2_PUBLIC_IP):
```ini
[Unit]
Description=Vitality Leisure Park Flask App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/vitality
Environment="COHERE_API_KEY=your-actual-key-here"
ExecStart=/home/ubuntu/vitality/venv/bin/gunicorn --workers 2 --bind 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable vitality
sudo systemctl start vitality
sudo systemctl status vitality  # Should show "active (running)"
```

---

## Step 7 — Set up Nginx (port 80 → 5000)

```bash
sudo nano /etc/nginx/sites-available/vitality
```

Paste:
```nginx
server {
    listen 80;
    server_name YOUR_EC2_PUBLIC_IP;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }

    location /static/ {
        alias /home/ubuntu/vitality/static/;
        expires 7d;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/vitality /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t  # Should say "test is successful"
sudo systemctl restart nginx
```

---

## Step 8 — Access your app

Open browser: **http://YOUR_EC2_PUBLIC_IP**

Your app is live!

---

## Useful commands

```bash
# View app logs
sudo journalctl -u vitality -f

# Restart app after code changes
sudo systemctl restart vitality

# Upload updated files
scp -i ~/Downloads/your-key.pem templates/wellness.html \
  ubuntu@YOUR_EC2_PUBLIC_IP:/home/ubuntu/vitality/templates/
sudo systemctl restart vitality
```

---

## Optional: Add a domain name

1. Buy a domain (Namecheap, GoDaddy, etc.)
2. Point DNS A record to your EC2 IP
3. Install SSL with Let's Encrypt:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

This gives you **https://yourdomain.com** for free.

---

## Cost estimate
- t2.micro: ~$0/month (free tier for 12 months, then ~$8/month)
- Static IP (Elastic IP): Free while instance running
- Storage: Included in free tier
