SELECT
    f.name as field_name,
    COUNT(*) as university_count,
    MIN(r.rank) as best_rank,
    AVG(r.rank) as average_rank,
    MAX(r.rank) as worst_rank,
    SUM(r.cites) as total_cites
FROM rankings r
JOIN universities u ON r.university_id = u.id
JOIN fields f ON r.field_id = f.id
WHERE u.country = 'CHINA MAINLAND'
GROUP BY f.name
ORDER BY f.name;