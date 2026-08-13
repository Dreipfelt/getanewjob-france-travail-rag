# GetANewJob — RAG multilingue pour offres d'emploi data

Système de recherche sémantique et de scoring qui compare un profil (CV) à un corpus d'offres d'emploi data, en s'appuyant sur l'API officielle France Travail.

Projet de démonstration technique (embeddings, vectorisation, retrieval, scoring LLM) construit pour combler une lacune récurrente identifiée dans plusieurs candidatures.

## Source des données

**API France Travail — Offres d'emploi v2**, accessible via [francetravail.io](https://francetravail.io). Accès libre sous réserve du respect de la [licence de réutilisation de la base de données des offres d'emploi de France Travail](https://francetravail.io/produits-partages/documentation/conditions-dutilisation-api/licence-offres-emploi).

Source : France Travail (francetravail.fr). Dernière extraction des données de démonstration : voir horodatage dans `data_raw/`.

## ⚠️ Obligations de la licence — à respecter impérativement

Ce projet réutilise des données soumises à une licence spécifique. Points de vigilance :

- **Pas de redistribution des données brutes.** L'article 3 de la licence interdit de mettre la base de données à disposition de tiers. **Le contenu de `data_raw/*.json` et le dump de la base PostgreSQL ne sont jamais commités ni publiés** — voir `.gitignore`. Seul le code (scripts, schéma SQL, prompts) est public.
- **Mention de source obligatoire.** Toute offre affichée (démo, capture d'écran, sortie d'un script) doit mentionner la source (France Travail) et la date de dernière mise à jour, avec un lien vers la licence — voir ce README.
- **Fraîcheur des données (article 5.2).** En usage continu, la licence impose d'interroger l'API au minimum une fois toutes les 24h et de répercuter les suppressions/modifications. **Ce projet est une démonstration ponctuelle** : les données extraites sont figées à un instant T (voir horodatage), pas un service à jour en continu. Pour tout usage en production, ce point devra être traité.
- **Usage conforme à la finalité.** Le projet sert à rapprocher une offre d'un profil candidat (rapprochement offre/demande d'emploi) — finalité alignée avec l'article 8 de la licence. Aucune constitution de fichier commercial, aucune revente.
- **Pas d'usage des logos** des entreprises/partenaires sans accord exprès (article 9) — non utilisés dans ce projet.

## Architecture

```
France Travail API (OAuth2 client_credentials)
        │
        ▼
  ingestion/ingest_offres.py  ──►  data_raw/offres_latest.json  (extraction multi-mots-clés, dédoublonnée)
        │
        ▼
  ingestion/sync_db.py         ──►  PostgreSQL + pgvector    (upsert incrémental, suppression des offres
        │                                                    disparues — conformité licence France Travail)
        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  api/api.py (FastAPI)                                          │
  │    POST /search  → recherche sémantique + filtres, gratuite   │
  │    POST /score   → scoring LLM motivé (Mistral), avec cache   │
  │    GET  /departements → départements disponibles en base      │
  └─────────────────────────────────────────────────────────────┘
        │
        ▼
  app/app.py (Streamlit)  ──►  interface utilisateur (profil, filtres, résultats, scoring à la demande)


  Orchestration (Airflow, DAG quotidien) :
  ingestion/ingest_offres.py  >>  ingestion/sync_db.py
  (exécuté dans un conteneur Airflow personnalisé, connecté au réseau
   Docker de la base pgvector)
```

Deux modes d'usage coexistent :
- **CLI direct** (`ingestion/ingest_offres.py`, `ingestion/sync_db.py`, `search/search.py`, `search/score_offres.py`) — pour du test manuel, avec `.env` pointant vers `PG_HOST=localhost`.
- **Orchestré** (Airflow, `airflow/dags/getanewjob_ingestion_dag.py`) — ingestion quotidienne automatique, avec `PG_HOST` injecté dynamiquement (`getanewjob-postgres`) indépendamment du `.env`.

## Stack

- **Ingestion** : API France Travail (OAuth2 client_credentials)
- **Stockage** : PostgreSQL + pgvector (Docker), index HNSW (similarité cosinus)
- **Embeddings** : `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768 dim, multilingue FR/NL/DE)
- **Scoring** : API Mistral (`mistral-small-latest`)
- **Backend** : FastAPI (`/search`, `/score`, `/departements`)
- **Interface** : Streamlit (profil en texte libre, filtres multi-sélection, scoring à la demande)
- **Orchestration** : Apache Airflow 3.3.1 (Docker Compose, image personnalisée avec dépendances du projet), DAG quotidien
- **Environnement** : Python 3.10, Docker Compose

## Prérequis

- Docker + Docker Compose
- Python 3.10+
- Un compte [francetravail.io](https://francetravail.io) avec une application souscrite à l'API "Offres d'emploi v2"
- Une clé API [Mistral](https://console.mistral.ai) (tier gratuit suffisant pour ce volume d'usage)

## Installation

```bash
pip install requests python-dotenv sentence-transformers psycopg2-binary pgvector mistralai fastapi uvicorn streamlit
docker compose up -d
```

Créer un fichier `.env` (voir `.env.example`) avec :
```
FT_CLIENT_ID=...
FT_CLIENT_SECRET=...
PG_HOST=localhost
PG_PORT=5432
PG_DB=getanewjob
PG_USER=getanewjob
PG_PASSWORD=...
MISTRAL_API_KEY=...
```

## Usage

### Lancement rapide

```bash
./start.sh
```

Démarre PostgreSQL/pgvector, Airflow, l'API FastAPI et Streamlit en une seule commande. Testé sous Linux avec GNOME Terminal (ouvre l'API et Streamlit dans de nouveaux onglets) ; sur un autre environnement, le script affiche les commandes à lancer manuellement en repli.

### En CLI (test manuel)

```bash
# 1. Extraire les offres depuis l'API France Travail
python ingestion/ingest_offres.py

# 2. Synchroniser la base (upsert + suppression des offres disparues)
python ingestion/sync_db.py

# 3. Recherche sémantique simple
python search/search.py mon_profil.md --type-contrat CDI --experience D

# 4. Scoring motivé par LLM sur le top-N
python search/score_offres.py mon_profil.md --type-contrat CDI --experience D --top-n 10
```

### Via l'interface web

```bash
# Terminal 1 — backend (depuis le dossier api/)
cd api && uvicorn api:app --reload --port 8000

# Terminal 2 — interface (depuis le dossier app/)
cd app && streamlit run app.py
```

Ouvrir `http://localhost:8501`, coller un profil, filtrer, rechercher, puis lancer le scoring motivé à la demande sur les résultats affichés.

### Orchestration automatisée (Airflow)

```bash
cd airflow
docker compose up -d
```

Interface Airflow sur `http://localhost:8090` (ou le port configuré si 8080 est déjà pris par un autre projet). Le DAG `getanewjob_ingestion_quotidienne` (ingestion + synchronisation) tourne quotidiennement, conformément à l'obligation de fraîcheur de la licence France Travail (article 5.2).

## État du projet

**Fonctionnel et testé de bout en bout :**
- Authentification OAuth2 et ingestion multi-mots-clés avec pagination et dédoublonnage
- Stockage vectoriel PostgreSQL/pgvector, avec synchronisation incrémentale (upsert + suppression des offres disparues, conforme à la licence de réutilisation)
- Recherche sémantique avec filtres structurés multi-sélection (contrat, département, expérience)
- Scoring LLM motivé (score + points forts/faibles + red flags), avec cache pour maîtriser les coûts
- Backend FastAPI et interface Streamlit, scoring déclenché à la demande pour ne pas exposer de coût caché
- Orchestration Airflow (DAG quotidien ingestion → synchronisation), avec image Docker personnalisée intégrant les dépendances du projet

**Non implémenté à ce stade :**
- Le prompt de scoring a été testé sur des échantillons de taille modeste (jusqu'à ~20 offres par run), pas à très grande échelle
- Pas de vraie authentification/autorisation sur l'API FastAPI (usage local uniquement)
- L'orchestration Airflow ne couvre que l'ingestion/synchronisation ; la recherche et le scoring restent des requêtes à la demande via FastAPI, pas des tâches planifiées

## Licence du code

Le code de ce dépôt est distribué sous licence [MIT](LICENSE). Les données issues de l'API France Travail restent soumises à leur licence propre (voir section ci-dessus) et ne sont pas couvertes par la licence du code.
