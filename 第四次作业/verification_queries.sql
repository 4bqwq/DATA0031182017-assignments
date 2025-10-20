SELECT 'Fields' as table_name, COUNT(*) as record_count FROM fields
UNION ALL
SELECT 'Universities', COUNT(*) FROM universities
UNION ALL
SELECT 'Rankings', COUNT(*) FROM rankings;

SELECT f.name, COUNT(*) as unis_per_field
FROM fields f
LEFT JOIN rankings r ON f.id = r.field_id
GROUP BY f.name
ORDER BY unis_per_field DESC;

SELECT u.country, COUNT(*) as universities_count
FROM universities u
GROUP BY u.country
ORDER BY universities_count DESC
LIMIT 10;