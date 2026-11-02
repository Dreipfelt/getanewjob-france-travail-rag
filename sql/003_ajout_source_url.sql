-- Ajout de la source et de l'URL d'origine à chaque offre.
-- Les offres existantes sont rattachées à France Travail.
ALTER TABLE offres
    ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'france_travail',
    ADD COLUMN IF NOT EXISTS url TEXT;

CREATE INDEX IF NOT EXISTS idx_offres_source ON offres (source);
