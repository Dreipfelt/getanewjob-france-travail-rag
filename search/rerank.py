"""
Reclassement (reranking) : reclasse un pool de candidats issus de la
recherche hybride à l'aide d'un cross-encoder multilingue, exécuté
entièrement en local (aucun appel réseau, aucun coût, aucune donnée
transmise à un tiers).

Pourquoi un cross-encoder après la recherche hybride plutôt qu'à sa place :
un cross-encoder score conjointement la paire (profil, offre) et est donc
plus précis qu'une comparaison de deux vecteurs indépendants (bi-encoder),
mais aussi bien plus lent — infaisable sur l'ensemble de la table. On
l'applique donc seulement sur un pool déjà réduit par la recherche hybride
(récupérer large, reclasser précisément, ne garder que le top N).

Contraintes du projet : le système doit rester gratuit et respecter le
RGPD (Mistral, une entreprise française, a été choisi pour le scoring
motivé pour cette raison). Le reranking tourne donc en local via
sentence-transformers, comme le modèle d'embedding déjà utilisé : mêmes
garanties (aucune donnée envoyée à un tiers), coût marginal nul.

search_and_rerank() compose hybrid_search() et rerank() pour les trois
points d'appel (search.py, score_offres.py, api.py), afin de ne pas
dupliquer trois fois le calcul de la taille du pool de candidats.
"""

from sentence_transformers import CrossEncoder

from hybrid_search import hybrid_search

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Taille du pool de candidats soumis au reranker, avant de ne garder que le
# top N final. Sur-échantillonner laisse au reranker de quoi arbitrer :
# une offre bien classée par RRF mais peu pertinente peut être écartée, une
# offre pertinente mais mal classée par RRF peut remonter.
POOL_MULTIPLIER = 5
POOL_MIN = 50

_reranker = None


def get_reranker() -> CrossEncoder:
    """Chargement paresseux et unique (modèle coûteux à charger)."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def rerank(profil_texte: str, offres: list, top_n: int) -> list:
    """
    Reclasse `offres` (sortie de hybrid_search, doit contenir la clé
    texte_embedding) par pertinence décroissante au profil, et retourne les
    `top_n` meilleures. Ajoute la clé score_rerank ; les clés score_rrf /
    rang_vectoriel / rang_motscles de hybrid_search sont conservées pour
    comparaison/debug.
    """
    if not offres:
        return []

    paires = [(profil_texte, offre["texte_embedding"]) for offre in offres]
    scores = get_reranker().predict(paires)

    for offre, score in zip(offres, scores):
        offre["score_rerank"] = float(score)

    offres_triees = sorted(offres, key=lambda o: o["score_rerank"], reverse=True)
    return offres_triees[:top_n]


def search_and_rerank(profil_texte, embedding_profil, *, limit=10,
                       with_description=False, **filtres_hybrid_search):
    """
    Pipeline complet : recherche hybride sur un pool élargi
    (max(limit * POOL_MULTIPLIER, POOL_MIN)), puis reclassement par
    cross-encoder pour ne garder que les `limit` meilleures offres.
    """
    pool_size = max(limit * POOL_MULTIPLIER, POOL_MIN)
    candidats = hybrid_search(
        profil_texte, embedding_profil,
        limit=pool_size, with_description=with_description,
        **filtres_hybrid_search,
    )
    return rerank(profil_texte, candidats, top_n=limit)
