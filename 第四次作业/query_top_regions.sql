WITH regional_performance AS (
    SELECT
        f.name as field_name,
        u.country,
        COUNT(*) as university_count,
        AVG(r.rank) as avg_rank,
        MIN(r.rank) as best_rank,
        MAX(r.cites) as max_cites,
        AVG(r.cites_per_paper) as avg_cites_per_paper,
        ROW_NUMBER() OVER (PARTITION BY f.name ORDER BY AVG(r.rank) ASC) as rank_in_field
    FROM rankings r
    JOIN universities u ON r.university_id = u.id
    JOIN fields f ON r.field_id = f.id
    GROUP BY f.name, u.country
    HAVING COUNT(*) >= 5
)
SELECT
    field_name,
    country,
    university_count,
    ROUND(avg_rank::NUMERIC, 1) as avg_rank,
    best_rank,
    avg_cites_per_paper
FROM regional_performance
WHERE rank_in_field <= 5
ORDER BY field_name, rank_in_field;