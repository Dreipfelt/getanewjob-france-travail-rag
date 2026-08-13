# Étend l'image officielle Airflow avec les dépendances nécessaires
# aux scripts du projet (ingest_offres.py, sync_db.py).
#
# Garder la version alignée avec celle utilisée dans docker-compose.yaml
# (apache/airflow:3.3.1 au moment de la rédaction).
FROM apache/airflow:3.3.1

# Les paquets pip doivent être installés en tant qu'utilisateur airflow,
# pas root (root échoue avec une erreur explicite côté Airflow).
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
