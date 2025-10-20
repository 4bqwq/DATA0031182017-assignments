#!/bin/bash

# CSV文件导入脚本

CONTAINER_NAME="lab4-postgres"
DATABASE="rankings"
USER="postgres"

declare -A FIELD_MAP=(
    ["AGRICULTURAL SCIENCES.csv"]="1"
    ["BIOLOGY & BIOCHEMISTRY.csv"]="2"
    ["CHEMISTRY.csv"]="3"
    ["CLINICAL MEDICINE.csv"]="4"
    ["COMPUTER SCIENCE.csv"]="5"
    ["ECONOMICS & BUSINESS.csv"]="6"
    ["ENGINEERING.csv"]="7"
    ["ENVIRONMENT ECOLOGY.csv"]="8"
    ["GEOSCIENCES.csv"]="9"
    ["IMMUNOLOGY.csv"]="10"
    ["MATERIALS SCIENCE.csv"]="11"
    ["MATHEMATICS.csv"]="12"
    ["MICROBIOLOGY.csv"]="13"
    ["MOLECULAR BIOLOGY & GENETICS.csv"]="14"
    ["MULTIDISCIPLINARY.csv"]="15"
    ["NEUROSCIENCE & BEHAVIOR.csv"]="16"
    ["PHARMACOLOGY & TOXICOLOGY.csv"]="17"
    ["PHYSICS.csv"]="18"
    ["PLANT & ANIMAL SCIENCE.csv"]="19"
    ["PSYCHIATRY PSYCHOLOGY.csv"]="20"
    ["SOCIAL SCIENCES, GENERAL.csv"]="21"
    ["SPACE SCIENCE.csv"]="22"
)

for csv_file in download/*.csv; do
    if [[ ! -f "$csv_file" ]]; then
        continue
    fi

    filename=$(basename "$csv_file")
    echo "Processing: $filename"

    field_id=${FIELD_MAP[$filename]}
    if [[ -z "$field_id" ]]; then
        echo "Warning: No field mapping for $filename, skipping..."
        continue
    fi

    docker cp "$csv_file" "$CONTAINER_NAME:/tmp/$filename"

    docker exec $CONTAINER_NAME bash -c "
        tail -n +3 '/tmp/$filename' | \
        sed '/^Copyright/d' | \
        sed '/^\s*$/d' > '/tmp/clean_$filename'
    "

    docker exec -i $CONTAINER_NAME psql -U $USER -d $DATABASE << EOF
DROP TABLE IF EXISTS temp_import CASCADE;
CREATE TABLE temp_import (
    rank_col VARCHAR(255),
    institution VARCHAR(255),
    country_region VARCHAR(255),
    documents VARCHAR(255),
    cites VARCHAR(255),
    cites_per_paper VARCHAR(255),
    top_papers VARCHAR(255)
);

psql -U $USER -d $DATABASE -c \"\\copy temp_import FROM '/tmp/clean_$filename' WITH CSV QUOTE '\\\"' ENCODING 'UTF-8';\"

INSERT INTO universities (name, country)
SELECT DISTINCT institution, country_region
FROM temp_import
WHERE institution IS NOT NULL AND institution != ''
ON CONFLICT (name) DO NOTHING;

INSERT INTO rankings (university_id, field_id, rank, documents, cites, cites_per_paper, top_papers)
SELECT
    u.id,
    $field_id,
    rank_col::INTEGER,
    documents::INTEGER,
    cites::INTEGER,
    cites_per_paper::DECIMAL(10,2),
    top_papers::INTEGER
FROM temp_import t
JOIN universities u ON u.name = t.institution
WHERE t.institution IS NOT NULL AND t.institution != '';

DROP TABLE temp_import;

SELECT 'Field $field_id ($filename): ' || COUNT(*) as imported_records
FROM rankings
WHERE field_id = $field_id;
EOF

    echo "Completed: $filename"
    echo "---"
done

echo "All CSV files imported successfully!"