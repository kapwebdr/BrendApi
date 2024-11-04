#!/bin/bash

# Attente des services
wait_for_service() {
    local service=$1
    local port=$2
    echo "Waiting for $service to be ready..."
    while ! nc -z $service $port; do
        sleep 1
    done
    echo "$service is ready!"
}

# Installation de curl et netcat dans les conteneurs
apt-get update && apt-get install -y curl netcat

# Attente des services
wait_for_service redis 6379
wait_for_service llm 8000
wait_for_service image_service 8000
wait_for_service translation 8000
wait_for_service speech 8000
wait_for_service media 8000
wait_for_service monitor 8000

# Démarrage de l'application
exec "$@" 