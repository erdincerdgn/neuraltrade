#!/bin/bash
# Generate self-signed SSL certificates for development
# For production, use Let's Encrypt with certbot

SSL_DIR="./nginx/ssl"
mkdir -p $SSL_DIR

# Generate private key
openssl genrsa -out $SSL_DIR/privkey.pem 2048

# Generate self-signed certificate
openssl req -new -x509 -key $SSL_DIR/privkey.pem -out $SSL_DIR/fullchain.pem -days 365 \
    -subj "//C=US/ST=State/L=City/O=NeuralTrade/CN=localhost"

echo "SSL certificates generated in $SSL_DIR"
echo "For production, replace with Let's Encrypt certificates"
