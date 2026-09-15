-- Migration : ajout de la recherche plein texte pour la recherche hybride
-- (vectorielle + mots-clés, fusionnées par Reciprocal Rank Fusion).
--
-- À exécuter sur une base déjà provisionnée par une version antérieure de
-- schema.sql. Une base recréée à partir du schema.sql à jour (volume Docker
-- neuf) n'a pas besoin de cette migration : la colonne y est déjà définie.
--
-- Usage :
--   docker exec -i getanewjob-postgres psql -U getanewjob -d getanewjob \
--       < sql/002_ajout_recherche_hybride.sql

ALTER TABLE offres
    ADD COLUMN IF NOT EXISTS texte_embedding_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('french', texte_embedding)) STORED;

CREATE INDEX IF NOT EXISTS idx_offres_texte_fts
    ON offres
    USING GIN (texte_embedding_tsv);
