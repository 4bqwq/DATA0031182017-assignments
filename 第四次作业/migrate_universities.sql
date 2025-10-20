INSERT INTO universities (name, country)
SELECT DISTINCT institution, country_region
FROM staging_rankings
WHERE institution IS NOT NULL AND institution != ''
ON CONFLICT (name) DO NOTHING;