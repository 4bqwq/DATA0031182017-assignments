INSERT INTO rankings (university_id, field_id, rank, documents, cites, cites_per_paper, top_papers)
SELECT
    u.id,
    5,
    rank_col::INTEGER,
    documents::INTEGER,
    cites::INTEGER,
    cites_per_paper::DECIMAL(10,2),
    top_papers::INTEGER
FROM staging_rankings s
JOIN universities u ON u.name = s.institution;