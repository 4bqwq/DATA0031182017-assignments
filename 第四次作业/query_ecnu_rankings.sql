SELECT
    f.name as field_name,
    r.rank,
    r.documents,
    r.cites,
    r.cites_per_paper,
    r.top_papers
FROM rankings r
JOIN universities u ON r.university_id = u.id
JOIN fields f ON r.field_id = f.id
WHERE u.name = 'EAST CHINA NORMAL UNIVERSITY'
ORDER BY r.rank;