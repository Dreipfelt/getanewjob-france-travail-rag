"""
Interface Streamlit pour GetANewJob.

Appelle l'API FastAPI (api.py) en HTTP — nécessite que celle-ci tourne
en parallèle (uvicorn api:app --reload --port 8000).

Lancement :
    streamlit run app.py
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="GetANewJob", page_icon="🔍", layout="wide")

st.title("GetANewJob")
st.caption(
    "Recherche sémantique et scoring motivé d'offres d'emploi data, "
    "à partir de l'API France Travail."
)
st.caption(
    "Source des données : France Travail (francetravail.fr). "
    "Réutilisation soumise à la "
    "[licence de réutilisation](https://francetravail.io/produits-partages/documentation/conditions-dutilisation-api/licence-offres-emploi)."
)

# --- Profil ---
st.subheader("Votre profil")
profil = st.text_area(
    "Collez le contenu de votre CV ou une description de votre profil",
    height=200,
    placeholder="Ex : 10 ans d'expérience en Data Engineering, maîtrise de Python, Airflow, PostgreSQL...",
)

# --- Filtres ---
st.subheader("Filtres (optionnels)")

CONTRAT_LABELS = {
    "CDI": "CDI",
    "CDD": "CDD",
    "MIS": "Intérim",
    "LIB": "Profession libérale",
}

@st.cache_data(ttl=3600)
def get_departements_disponibles():
    try:
        resp = requests.get(f"{API_URL}/departements", timeout=10)
        resp.raise_for_status()
        return resp.json()["departements"]
    except requests.exceptions.RequestException:
        return []

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Type de contrat**")
    types_contrat_selectionnes = [
        code for code, label in CONTRAT_LABELS.items()
        if st.checkbox(label, key=f"contrat_{code}")
    ]

with col2:
    st.markdown("**Département**")
    departements_disponibles = get_departements_disponibles()
    departements_selectionnes = st.multiselect(
        "Un ou plusieurs départements",
        options=departements_disponibles,
        label_visibility="collapsed",
    )

col3, col4 = st.columns(2)
with col3:
    experience = st.selectbox(
        "Expérience", ["", "D", "E"],
        format_func=lambda x: {"": "Indifférent", "D": "Débutant accepté", "E": "Expérience exigée"}[x],
    )
with col4:
    limit = st.number_input("Nombre de résultats", min_value=1, max_value=50, value=20)


def build_payload():
    return {
        "profil": profil,
        "types_contrat": types_contrat_selectionnes or None,
        "departements": departements_selectionnes or None,
        "experience": experience or None,
    }


# --- Recherche sémantique ---
if st.button("🔍 Rechercher (gratuit)", type="primary", disabled=not profil.strip()):
    with st.spinner("Recherche en cours..."):
        try:
            payload = {**build_payload(), "limit": limit}
            resp = requests.post(f"{API_URL}/search", json=payload, timeout=30)
            resp.raise_for_status()
            resultats = resp.json()["resultats"]
            st.session_state["resultats_recherche"] = resultats
            st.session_state.pop("resultats_score", None)  # invalide un scoring précédent
        except requests.exceptions.ConnectionError:
            st.error(
                "Impossible de joindre l'API. Vérifiez qu'elle tourne bien "
                "(`uvicorn api:app --reload --port 8000`)."
            )
        except requests.exceptions.HTTPError as e:
            st.error(f"Erreur de l'API : {e}")

# --- Affichage des résultats de recherche ---
if "resultats_recherche" in st.session_state:
    resultats = st.session_state["resultats_recherche"]

    if not resultats:
        st.info("Aucune offre ne correspond à vos critères.")
    else:
        st.subheader(f"{len(resultats)} offres trouvées")

        for r in resultats:
            with st.container(border=True):
                st.markdown(f"**{r['intitule']}**")
                st.caption(
                    f"{r.get('entreprise_nom') or 'Entreprise non précisée'} · "
                    f"{r.get('lieu_libelle') or 'Lieu non précisé'} · "
                    f"{r.get('type_contrat_libelle') or 'Contrat non précisé'} · "
                    f"{r.get('experience_libelle') or 'Expérience non précisée'}"
                )
                st.caption(f"Similarité : {1 - r['distance']:.2%} · id: {r['id']}")

        st.divider()

        # --- Scoring LLM à la demande, uniquement sur ce qui a été trouvé ---
        st.subheader("Scoring motivé")
        st.caption(
            "Analyse plus poussée de chaque offre par rapport à votre profil, "
            "via un appel à l'API Mistral. Ceci a un coût réel par offre non "
            "encore analysée (les offres déjà scorées sont servies depuis le cache)."
        )

        if st.button("🎯 Lancer le scoring motivé sur ces offres"):
            with st.spinner("Analyse en cours (peut prendre du temps selon le nombre d'offres)..."):
                try:
                    payload = {**build_payload(), "top_n": limit}
                    resp = requests.post(f"{API_URL}/score", json=payload, timeout=180)
                    resp.raise_for_status()
                    data = resp.json()
                    st.session_state["resultats_score"] = data["resultats"]
                    st.success(
                        f"{data['appels_api']} appel(s) API effectué(s), "
                        f"{data['servis_depuis_cache']} servi(s) depuis le cache."
                    )
                except requests.exceptions.ConnectionError:
                    st.error("Impossible de joindre l'API.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"Erreur de l'API : {e}")

# --- Affichage des résultats scorés ---
if "resultats_score" in st.session_state:
    st.subheader("Résultats triés par score de pertinence")

    for r in st.session_state["resultats_score"]:
        with st.container(border=True):
            col_titre, col_score = st.columns([4, 1])
            with col_titre:
                st.markdown(f"**{r['intitule']}**")
                st.caption(f"{r.get('entreprise_nom') or 'N/A'} · {r.get('lieu_libelle') or 'N/A'}")
            with col_score:
                st.metric("Score", f"{r['score']}/100")

            if r.get("points_forts"):
                st.markdown("**Points forts**")
                for p in r["points_forts"]:
                    st.markdown(f"- {p}")

            if r.get("points_faibles"):
                st.markdown("**Points faibles**")
                for p in r["points_faibles"]:
                    st.markdown(f"- {p}")

            if r.get("red_flags"):
                st.markdown("**⚠️ Signaux d'alerte**")
                for p in r["red_flags"]:
                    st.markdown(f"- {p}")
