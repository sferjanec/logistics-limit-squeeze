-- This script runs automatically the first time the container boots up.

-- 1. Enable the pgvector extension in our specific database
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. (Optional but professional) Create an explicit schema for our AI agents
CREATE SCHEMA IF NOT EXISTS agent_memory;

-- 3. Set the search path so our tables default to this schema
ALTER ROLE postgres SET search_path TO agent_memory, public;