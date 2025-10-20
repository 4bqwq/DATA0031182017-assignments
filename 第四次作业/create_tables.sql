-- 创建学科表
CREATE TABLE fields (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

-- 创建大学表
CREATE TABLE universities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    country VARCHAR(100) NOT NULL
);

-- 创建排名表
CREATE TABLE rankings (
    id SERIAL PRIMARY KEY,
    university_id INTEGER REFERENCES universities(id),
    field_id INTEGER REFERENCES fields(id),
    rank INTEGER NOT NULL,
    documents INTEGER,
    cites INTEGER,
    cites_per_paper DECIMAL(10,2),
    top_papers INTEGER,
    UNIQUE(university_id, field_id)
);

-- 创建索引
CREATE INDEX idx_university_country ON universities(country);
CREATE INDEX idx_rankings_university ON rankings(university_id);
CREATE INDEX idx_rankings_field ON rankings(field_id);