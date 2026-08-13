"""
DAG d'ingestion quotidienne des offres France Travail.

Orchestre :
1. ingest_offres.py  -> récupère les offres actuelles depuis l'API
2. sync_db.py        -> upsert des offres nouvelles/modifiées, suppression
                         des offres disparues (conformité article 5.2 de la
                         licence de réutilisation France Travail)

Fréquence : quotidienne, conformément à l'obligation de fraîcheur minimale
de la licence (interroger l'API au moins une fois toutes les 24h).
"""

from datetime import datetime, timedelta
from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_DIR = "/opt/getanewjob"

default_args = {
    "owner": "getanewjob",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="getanewjob_ingestion_quotidienne",
    description="Ingestion France Travail + synchronisation PostgreSQL/pgvector",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 8, 13),
    catchup=False,
    tags=["getanewjob", "france-travail"],
) as dag:

    ingest = BashOperator(
        task_id="ingest_offres",
        bash_command=f"cd {PROJECT_DIR} && python ingestion/ingest_offres.py",
    )

    sync = BashOperator(
        task_id="sync_db",
        bash_command=f"cd {PROJECT_DIR} && python ingestion/sync_db.py",
        env={"PG_HOST": "getanewjob-postgres"},
        append_env=True,
    )

    ingest >> sync
