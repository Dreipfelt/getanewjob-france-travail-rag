#!/usr/bin/env bash
#
# Lance l'intégralité de l'environnement GetANewJob :
#   1. PostgreSQL/pgvector (Docker, arrière-plan)
#   2. Airflow (Docker Compose, arrière-plan)
#   3. FastAPI (nouvel onglet GNOME Terminal)
#   4. Streamlit (nouvel onglet GNOME Terminal)
#
# Usage :
#   ./start.sh
#
# Prérequis : Docker, gnome-terminal, environnement Python avec les
# dépendances de requirements.txt déjà installées.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=== GetANewJob — démarrage de l'environnement ==="
echo

# --- 1. PostgreSQL / pgvector ---
echo "[1/4] Démarrage de PostgreSQL/pgvector..."
docker compose up -d
echo "  -> OK"
echo

# --- 2. Airflow ---
echo "[2/4] Démarrage d'Airflow..."
if [ -d "airflow" ]; then
    (cd airflow && docker compose up -d)
    echo "  -> OK (interface sur le port configuré dans airflow/docker-compose.yaml,"
    echo "     ex. http://localhost:8090 — vérifiez le mapping de port si conflit)"
else
    echo "  -> ATTENTION : dossier airflow/ introuvable, étape ignorée."
fi
echo

# Petite pause pour laisser PostgreSQL être prêt à accepter des connexions
# avant que l'API ne démarre et tente de s'y connecter.
echo "Attente de 5 secondes pour la disponibilité de PostgreSQL..."
sleep 5
echo

# --- 3. FastAPI dans un nouvel onglet ---
echo "[3/4] Lancement de l'API FastAPI (nouvel onglet)..."
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal --tab --title="GetANewJob - API" -- bash -c \
        "cd '$PROJECT_ROOT/api' && uvicorn api:app --reload --port 8000; exec bash"
    echo "  -> OK, onglet ouvert"
else
    echo "  -> gnome-terminal introuvable. Lancez manuellement :"
    echo "     cd api && uvicorn api:app --reload --port 8000"
fi
echo

# --- 4. Streamlit dans un nouvel onglet ---
echo "[4/4] Lancement de Streamlit (nouvel onglet)..."
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal --tab --title="GetANewJob - Streamlit" -- bash -c \
        "cd '$PROJECT_ROOT/app' && streamlit run app.py; exec bash"
    echo "  -> OK, onglet ouvert"
else
    echo "  -> gnome-terminal introuvable. Lancez manuellement :"
    echo "     cd app && streamlit run app.py"
fi
echo

echo "=== Tout est lancé ==="
echo "Interface Streamlit : http://localhost:8501"
echo "API FastAPI          : http://localhost:8000/docs"
echo "Interface Airflow    : voir port configuré dans airflow/docker-compose.yaml"
