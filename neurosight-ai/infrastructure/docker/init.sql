-- NeuroSight AI — PostgreSQL initialization
-- Runs once on first container start

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create test database (used by CI)
SELECT 'CREATE DATABASE neurosight_test'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'neurosight_test')\gexec

GRANT ALL PRIVILEGES ON DATABASE neurosight TO neurosight;
GRANT ALL PRIVILEGES ON DATABASE neurosight_test TO neurosight;
