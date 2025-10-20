DROP TABLE IF EXISTS staging_rankings CASCADE;

CREATE TABLE staging_rankings (
    rank_col VARCHAR(255),
    institution VARCHAR(255),
    country_region VARCHAR(255),
    documents VARCHAR(255),
    cites VARCHAR(255),
    cites_per_paper VARCHAR(255),
    top_papers VARCHAR(255)
);