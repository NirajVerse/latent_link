#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

IP=$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' | head -1)
[ -z "$IP" ] && IP=127.0.0.1

if [ ! -f ssl/cert.pem ]; then
  echo "Generating self-signed certificate..."
  mkdir -p ssl
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout ssl/key.pem -out ssl/cert.pem -days 365 \
    -subj "/CN=latentlink" \
    -addext "subjectAltName=IP:$IP,IP:127.0.0.1,DNS:localhost"
fi

echo ""
echo "=============================================================="
echo "  Sender (phone 1):   https://$IP:8000/static/sender.html"
echo "  Receiver (phone 2): https://$IP:8000/static/receiver.html"
echo "=============================================================="
echo ""

exec .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 \
  --ssl-keyfile ssl/key.pem --ssl-certfile ssl/cert.pem
