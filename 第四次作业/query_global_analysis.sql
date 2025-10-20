SELECT
    f.name as field_name,
    u.country,
    COUNT(*) as university_count,
    AVG(r.rank) as average_rank,
    MIN(r.rank) as best_rank,
    MAX(r.cites) as max_cites,
    AVG(r.cites_per_paper) as avg_cites_per_paper
FROM rankings r
JOIN universities u ON r.university_id = u.id
JOIN fields f ON r.field_id = f.id
GROUP BY f.name, u.country
HAVING COUNT(*) >= 10
ORDER BY f.name, average_rank;