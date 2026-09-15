-- Schéma pour le projet RAG offres d'emploi
-- Modèle d'embedding cible : sentence-transformers/paraphrase-multilingual-mpnet-base-v2
-- Dimension confirmée : 768 (source : model card officielle HuggingFace)

-- Prérequis : extension pgvector installée sur l'instance PostgreSQL
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE offres (
    -- Identifiant unique fourni par l'API France Travail (champ "id" confirmé
    -- présent et unique sur les 795 offres testées)
    id              TEXT PRIMARY KEY,

    -- Champs sources bruts, utiles à l'affichage et à la traçabilité
    intitule        TEXT NOT NULL,
    description     TEXT,

    -- Champs de filtrage structuré (colonnes classiques, pas vectorisés)
    entreprise_nom      TEXT,
    lieu_libelle         TEXT,
    lieu_code_postal     TEXT,
    lieu_latitude         DOUBLE PRECISION,
    lieu_longitude         DOUBLE PRECISION,
    type_contrat          TEXT,       -- ex. CDI, CDD
    type_contrat_libelle  TEXT,
    experience_exige       TEXT,       -- code : D/S/E (débutant/souhaité/exigé - à vérifier)
    experience_libelle     TEXT,
    rome_code               TEXT,
    rome_libelle             TEXT,
    qualification_code       TEXT,
    qualification_libelle     TEXT,

    date_creation        TIMESTAMPTZ,
    date_actualisation   TIMESTAMPTZ,

    -- Données peu structurantes pour filtrer, gardées en JSON brut plutôt que
    -- d'exploser en dizaines de colonnes peu utilisées (salaire, contact,
    -- compétences détaillées, qualités professionnelles)
    competences_json     JSONB,
    salaire_json          JSONB,
    contact_json           JSONB,
    raw_json                 JSONB,     -- offre complète en backup, au cas où un champ non prévu ici devienne utile plus tard

    -- Le texte réellement concaténé et envoyé au modèle d'embedding
    -- (titre + description + libellés compétences). Conservé pour audit /
    -- debug : si un résultat de recherche semble étrange, on peut relire
    -- exactement ce qui a été vectorisé.
    texte_embedding       TEXT NOT NULL,

    -- Vecteur plein texte (français), dérivé automatiquement de
    -- texte_embedding. Alimente la branche "mots-clés" de la recherche
    -- hybride (vectorielle + full-text, fusionnées par Reciprocal Rank
    -- Fusion) : capture les correspondances exactes de termes (techno,
    -- intitulé de poste) que la similarité sémantique seule peut diluer
    -- quand elle est noyée dans un profil long.
    texte_embedding_tsv   tsvector GENERATED ALWAYS AS (to_tsvector('french', texte_embedding)) STORED,

    -- Le vecteur lui-même
    embedding             vector(768),

    -- Traçabilité de l'ingestion
    date_extraction        TIMESTAMPTZ NOT NULL DEFAULT now(),
    mots_cles_recherche    TEXT[]      -- quels mots-clés de recherche ont remonté cette offre
);

-- Index vectoriel HNSW pour la recherche par similarité cosinus.
-- Note : la métrique de distance doit correspondre à celle utilisée au
-- moment de la requête (vector_cosine_ops <-> pour du cosinus). Si vous
-- utilisez une autre métrique (L2, produit scalaire), il faudra un index
-- différent (vector_l2_ops, vector_ip_ops).
CREATE INDEX idx_offres_embedding_hnsw
    ON offres
    USING hnsw (embedding vector_cosine_ops);

-- Index full-text (GIN) pour la branche mots-clés de la recherche hybride.
CREATE INDEX idx_offres_texte_fts
    ON offres
    USING GIN (texte_embedding_tsv);

-- Index classiques pour le filtrage pré/post-retrieval
CREATE INDEX idx_offres_type_contrat ON offres (type_contrat);
CREATE INDEX idx_offres_rome_code ON offres (rome_code);
CREATE INDEX idx_offres_date_creation ON offres (date_creation);
CREATE INDEX idx_offres_lieu_code_postal ON offres (lieu_code_postal);
