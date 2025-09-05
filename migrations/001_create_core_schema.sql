-- =====================================================
-- Phase 1.2: Advanced Database Architecture 
-- ARGO Oceanographic Data Schema for LLM Queries
-- =====================================================

-- Enable PostGIS extension for spatial operations
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS pg_trgm; -- For text search
CREATE EXTENSION IF NOT EXISTS btree_gist; -- For advanced indexing

-- =====================================================
-- Core Tables for ARGO Float Data
-- =====================================================

-- Data Assembly Centers (DACs) lookup table
CREATE TABLE dacs (
    id SERIAL PRIMARY KEY,
    code VARCHAR(10) UNIQUE NOT NULL, -- e.g., 'incois', 'aoml'
    name VARCHAR(255) NOT NULL,
    country VARCHAR(100),
    institution VARCHAR(255),
    contact_email VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ARGO Float registry with comprehensive metadata
CREATE TABLE floats (
    id SERIAL PRIMARY KEY,
    platform_number VARCHAR(20) UNIQUE NOT NULL, -- e.g., '2900226'
    dac_id INTEGER REFERENCES dacs(id),
    
    -- Physical characteristics
    float_serial_no VARCHAR(50),
    wmo_inst_type VARCHAR(10),
    platform_type VARCHAR(50),
    platform_maker VARCHAR(100),
    firmware_version VARCHAR(50),
    
    -- Deployment information
    deployment_date DATE,
    deployment_latitude DOUBLE PRECISION,
    deployment_longitude DOUBLE PRECISION,
    deployment_geom GEOMETRY(POINT, 4326), -- PostGIS geometry
    
    -- Mission details
    project_name VARCHAR(255),
    pi_name VARCHAR(255),
    data_centre VARCHAR(50),
    
    -- Status tracking
    is_active BOOLEAN DEFAULT TRUE,
    last_profile_date DATE,
    total_profiles INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Spatial index on deployment location
    CONSTRAINT valid_deployment_coords CHECK (
        deployment_latitude BETWEEN -90 AND 90 AND
        deployment_longitude BETWEEN -180 AND 180
    )
);

-- ARGO Profile instances (individual measurements)
CREATE TABLE profiles (
    id SERIAL PRIMARY KEY,
    profile_uuid UUID DEFAULT gen_random_uuid(), -- Unique identifier
    float_id INTEGER REFERENCES floats(id) ON DELETE CASCADE,
    
    -- Profile identification
    cycle_number INTEGER NOT NULL,
    direction VARCHAR(1), -- 'A' for ascending, 'D' for descending
    profile_filename VARCHAR(255), -- Original NetCDF filename
    
    -- Spatial-temporal information
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    geom GEOMETRY(POINT, 4326), -- PostGIS geometry for spatial queries
    
    -- Temporal information
    juld DOUBLE PRECISION, -- Julian day relative to 1950-01-01
    measurement_date TIMESTAMP WITH TIME ZONE NOT NULL,
    measurement_year INTEGER GENERATED ALWAYS AS (EXTRACT(YEAR FROM measurement_date)) STORED,
    measurement_month INTEGER GENERATED ALWAYS AS (EXTRACT(MONTH FROM measurement_date)) STORED,
    measurement_season INTEGER GENERATED ALWAYS AS (
        CASE 
            WHEN EXTRACT(MONTH FROM measurement_date) IN (12, 1, 2) THEN 1 -- Winter
            WHEN EXTRACT(MONTH FROM measurement_date) IN (3, 4, 5) THEN 2 -- Spring
            WHEN EXTRACT(MONTH FROM measurement_date) IN (6, 7, 8) THEN 3 -- Summer
            ELSE 4 -- Autumn
        END
    ) STORED,
    
    -- Data quality and processing
    data_mode VARCHAR(1), -- 'R' for real-time, 'D' for delayed-mode
    position_qc CHAR(1),
    vertical_sampling_scheme TEXT,
    
    -- Profile summary statistics (for LLM queries)
    max_depth_m DOUBLE PRECISION,
    min_temperature_c DOUBLE PRECISION,
    max_temperature_c DOUBLE PRECISION,
    min_salinity_psu DOUBLE PRECISION,
    max_salinity_psu DOUBLE PRECISION,
    valid_measurements_count INTEGER DEFAULT 0,
    
    -- Quality flags for entire profile
    profile_temp_qc CHAR(1),
    profile_psal_qc CHAR(1),
    profile_pres_qc CHAR(1),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(float_id, cycle_number),
    CONSTRAINT valid_coordinates CHECK (
        latitude BETWEEN -90 AND 90 AND
        longitude BETWEEN -180 AND 180
    ),
    CONSTRAINT valid_qc_flags CHECK (
        position_qc IN ('1', '2', '3', '4', '5', '8', '9') AND
        profile_temp_qc IN ('1', '2', '3', '4', '5', '8', '9') AND
        profile_psal_qc IN ('1', '2', '3', '4', '5', '8', '9') AND
        profile_pres_qc IN ('1', '2', '3', '4', '5', '8', '9')
    )
);

-- Individual depth measurements (normalized for fast queries)
CREATE TABLE measurements (
    id BIGSERIAL PRIMARY KEY,
    profile_id INTEGER REFERENCES profiles(id) ON DELETE CASCADE,
    
    -- Depth information
    depth_level INTEGER NOT NULL, -- Level index in NetCDF
    pressure_db REAL, -- Pressure in decibars
    depth_m REAL, -- Approximate depth in meters
    
    -- Oceanographic parameters
    temperature_c REAL, -- Temperature in Celsius
    salinity_psu REAL, -- Practical Salinity Units
    
    -- Adjusted values (delayed-mode processing)
    temperature_adjusted_c REAL,
    salinity_adjusted_c REAL,
    pressure_adjusted_db REAL,
    
    -- Quality control flags
    temperature_qc CHAR(1),
    salinity_qc CHAR(1),
    pressure_qc CHAR(1),
    temperature_adjusted_qc CHAR(1),
    salinity_adjusted_qc CHAR(1),
    pressure_adjusted_qc CHAR(1),
    
    -- Error estimates
    temperature_error REAL,
    salinity_error REAL,
    pressure_error REAL,
    
    -- Data validity flags
    is_valid BOOLEAN GENERATED ALWAYS AS (
        pressure_qc IN ('1', '2') AND 
        temperature_qc IN ('1', '2') AND 
        salinity_qc IN ('1', '2')
    ) STORED,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_measurements CHECK (
        pressure_db >= 0 AND
        temperature_c >= -3 AND temperature_c <= 50 AND
        salinity_psu >= 0 AND salinity_psu <= 50
    ),
    CONSTRAINT valid_qc_flags CHECK (
        temperature_qc IN ('1', '2', '3', '4', '5', '8', '9') AND
        salinity_qc IN ('1', '2', '3', '4', '5', '8', '9') AND
        pressure_qc IN ('1', '2', '3', '4', '5', '8', '9')
    )
);

-- =====================================================
-- LLM-Optimized Tables and Views
-- =====================================================

-- Regional classifications for natural language queries
CREATE TABLE ocean_regions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    aliases TEXT[], -- Alternative names for the region
    boundary GEOMETRY(POLYGON, 4326), -- Region boundary
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pre-computed statistics for fast LLM responses
CREATE TABLE profile_statistics (
    id SERIAL PRIMARY KEY,
    profile_id INTEGER REFERENCES profiles(id) ON DELETE CASCADE,
    
    -- Temperature statistics
    temp_mean REAL,
    temp_std REAL,
    temp_min REAL,
    temp_max REAL,
    temp_surface REAL, -- Temperature at shallowest depth
    temp_bottom REAL,  -- Temperature at deepest depth
    
    -- Salinity statistics  
    sal_mean REAL,
    sal_std REAL,
    sal_min REAL,
    sal_max REAL,
    sal_surface REAL,
    sal_bottom REAL,
    
    -- Depth statistics
    max_depth REAL,
    valid_depths_count INTEGER,
    
    -- Derived parameters
    mixed_layer_depth REAL,
    thermocline_depth REAL,
    
    -- Quality metrics
    data_quality_score REAL, -- Overall quality assessment
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Monthly aggregations for trend analysis
CREATE TABLE monthly_climatology (
    id SERIAL PRIMARY KEY,
    region_id INTEGER REFERENCES ocean_regions(id),
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    depth_range_start REAL NOT NULL,
    depth_range_end REAL NOT NULL,
    
    -- Aggregated measurements
    avg_temperature REAL,
    std_temperature REAL,
    avg_salinity REAL,
    std_salinity REAL,
    measurement_count INTEGER,
    
    -- Spatial coverage
    latitude_range NUMRANGE,
    longitude_range NUMRANGE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(region_id, year, month, depth_range_start, depth_range_end)
);

-- =====================================================
-- Indexes for Optimal LLM Query Performance
-- =====================================================

-- Spatial indexes
CREATE INDEX idx_floats_deployment_geom ON floats USING GIST(deployment_geom);
CREATE INDEX idx_profiles_geom ON profiles USING GIST(geom);
CREATE INDEX idx_profiles_spatial_temporal ON profiles USING GIST(geom, measurement_date);

-- Temporal indexes
CREATE INDEX idx_profiles_date ON profiles(measurement_date);
CREATE INDEX idx_profiles_year_month ON profiles(measurement_year, measurement_month);
CREATE INDEX idx_profiles_season ON profiles(measurement_season);

-- Float and profile relationships
CREATE INDEX idx_profiles_float_id ON profiles(float_id);
CREATE INDEX idx_profiles_cycle ON profiles(float_id, cycle_number);
CREATE INDEX idx_measurements_profile ON measurements(profile_id);

-- Oceanographic parameter indexes
CREATE INDEX idx_measurements_depth ON measurements(depth_m) WHERE is_valid;
CREATE INDEX idx_measurements_temp ON measurements(temperature_c) WHERE is_valid;
CREATE INDEX idx_measurements_sal ON measurements(salinity_psu) WHERE is_valid;

-- Quality control indexes
CREATE INDEX idx_profiles_qc ON profiles(data_mode, profile_temp_qc, profile_psal_qc);
CREATE INDEX idx_measurements_valid ON measurements(is_valid, pressure_qc, temperature_qc, salinity_qc);

-- Text search indexes for LLM queries
CREATE INDEX idx_floats_text_search ON floats USING GIN(to_tsvector('english', 
    COALESCE(project_name, '') || ' ' || COALESCE(pi_name, '')));

-- Composite indexes for common query patterns
CREATE INDEX idx_spatial_temporal_quality ON profiles USING GIST(
    geom, measurement_date, data_mode
);

-- =====================================================
-- Triggers for Automatic Updates
-- =====================================================

-- Update float statistics when profiles are added
CREATE OR REPLACE FUNCTION update_float_statistics()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE floats SET 
        total_profiles = (
            SELECT COUNT(*) FROM profiles WHERE float_id = NEW.float_id
        ),
        last_profile_date = (
            SELECT MAX(measurement_date::date) FROM profiles WHERE float_id = NEW.float_id
        ),
        updated_at = NOW()
    WHERE id = NEW.float_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_float_stats
    AFTER INSERT OR UPDATE ON profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_float_statistics();

-- Update profile statistics when measurements are added
CREATE OR REPLACE FUNCTION update_profile_statistics()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE profiles SET 
        valid_measurements_count = (
            SELECT COUNT(*) FROM measurements 
            WHERE profile_id = NEW.profile_id AND is_valid = TRUE
        ),
        max_depth_m = (
            SELECT MAX(depth_m) FROM measurements 
            WHERE profile_id = NEW.profile_id AND is_valid = TRUE
        ),
        min_temperature_c = (
            SELECT MIN(temperature_c) FROM measurements 
            WHERE profile_id = NEW.profile_id AND is_valid = TRUE
        ),
        max_temperature_c = (
            SELECT MAX(temperature_c) FROM measurements 
            WHERE profile_id = NEW.profile_id AND is_valid = TRUE
        ),
        min_salinity_psu = (
            SELECT MIN(salinity_psu) FROM measurements 
            WHERE profile_id = NEW.profile_id AND is_valid = TRUE
        ),
        max_salinity_psu = (
            SELECT MAX(salinity_psu) FROM measurements 
            WHERE profile_id = NEW.profile_id AND is_valid = TRUE
        ),
        updated_at = NOW()
    WHERE id = NEW.profile_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_profile_stats
    AFTER INSERT OR UPDATE ON measurements
    FOR EACH ROW
    EXECUTE FUNCTION update_profile_statistics();

-- Update geometry columns automatically
CREATE OR REPLACE FUNCTION update_geometry_columns()
RETURNS TRIGGER AS $$
BEGIN
    -- Update deployment geometry for floats
    IF TG_TABLE_NAME = 'floats' THEN
        NEW.deployment_geom = ST_SetSRID(ST_MakePoint(NEW.deployment_longitude, NEW.deployment_latitude), 4326);
    END IF;
    
    -- Update profile geometry
    IF TG_TABLE_NAME = 'profiles' THEN
        NEW.geom = ST_SetSRID(ST_MakePoint(NEW.longitude, NEW.latitude), 4326);
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_float_geom
    BEFORE INSERT OR UPDATE ON floats
    FOR EACH ROW
    EXECUTE FUNCTION update_geometry_columns();

CREATE TRIGGER trg_update_profile_geom
    BEFORE INSERT OR UPDATE ON profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_geometry_columns();

-- =====================================================
-- Initial Data Setup
-- =====================================================

-- Insert common DACs
INSERT INTO dacs (code, name, country, institution) VALUES
    ('incois', 'Indian National Centre for Ocean Information Services', 'India', 'Ministry of Earth Sciences'),
    ('aoml', 'Atlantic Oceanographic and Meteorological Laboratory', 'USA', 'NOAA'),
    ('bodc', 'British Oceanographic Data Centre', 'UK', 'National Oceanography Centre'),
    ('coriolis', 'Coriolis Operational Oceanography', 'France', 'Ifremer'),
    ('csio', 'Canadian Scientific Information Operations', 'Canada', 'Fisheries and Oceans Canada'),
    ('csiro', 'Commonwealth Scientific and Industrial Research Organisation', 'Australia', 'CSIRO'),
    ('jma', 'Japan Meteorological Agency', 'Japan', 'Ministry of Land, Infrastructure, Transport and Tourism'),
    ('meds', 'Marine Environmental Data Section', 'Canada', 'Fisheries and Oceans Canada');

-- Insert major ocean regions for LLM queries
INSERT INTO ocean_regions (name, aliases, description) VALUES
    ('Arabian Sea', ARRAY['Arabian Sea', 'northwestern Indian Ocean'], 
     'Northwestern region of the Indian Ocean bounded by India, Pakistan, Iran, and the Arabian Peninsula'),
    ('Bay of Bengal', ARRAY['Bay of Bengal', 'northeastern Indian Ocean'],
     'Northeastern region of the Indian Ocean bounded by India, Bangladesh, Myanmar, and Sri Lanka'),
    ('Indian Ocean', ARRAY['Indian Ocean'], 
     'Third largest ocean, bounded by Africa, Asia, Australia, and Antarctica'),
    ('North Atlantic', ARRAY['North Atlantic Ocean', 'Northern Atlantic'],
     'Northern region of the Atlantic Ocean'),
    ('South Atlantic', ARRAY['South Atlantic Ocean', 'Southern Atlantic'],
     'Southern region of the Atlantic Ocean'),
    ('North Pacific', ARRAY['North Pacific Ocean', 'Northern Pacific'],
     'Northern region of the Pacific Ocean'),
    ('South Pacific', ARRAY['South Pacific Ocean', 'Southern Pacific'],
     'Southern region of the Pacific Ocean'),
    ('Southern Ocean', ARRAY['Southern Ocean', 'Antarctic Ocean'],
     'Ocean surrounding Antarctica'),
    ('Mediterranean Sea', ARRAY['Mediterranean', 'Med Sea'],
     'Sea connected to the Atlantic Ocean and enclosed by the Mediterranean Basin');

-- Create helpful views for LLM queries
CREATE VIEW v_recent_profiles AS
SELECT 
    p.id,
    p.profile_uuid,
    f.platform_number,
    p.cycle_number,
    p.latitude,
    p.longitude,
    p.measurement_date,
    p.data_mode,
    p.max_depth_m,
    p.min_temperature_c,
    p.max_temperature_c,
    p.min_salinity_psu,
    p.max_salinity_psu,
    p.valid_measurements_count,
    d.name as dac_name
FROM profiles p
JOIN floats f ON p.float_id = f.id
JOIN dacs d ON f.dac_id = d.id
WHERE p.measurement_date >= NOW() - INTERVAL '1 year';

CREATE VIEW v_profile_summary AS
SELECT 
    p.id as profile_id,
    f.platform_number,
    p.cycle_number,
    p.latitude,
    p.longitude,
    p.measurement_date,
    p.measurement_year,
    p.measurement_season,
    p.data_mode,
    COUNT(m.id) as total_measurements,
    COUNT(m.id) FILTER (WHERE m.is_valid) as valid_measurements,
    MAX(m.depth_m) as max_depth,
    AVG(m.temperature_c) FILTER (WHERE m.is_valid) as avg_temperature,
    AVG(m.salinity_psu) FILTER (WHERE m.is_valid) as avg_salinity,
    MIN(m.temperature_c) FILTER (WHERE m.is_valid) as min_temperature,
    MAX(m.temperature_c) FILTER (WHERE m.is_valid) as max_temperature
FROM profiles p
JOIN floats f ON p.float_id = f.id
LEFT JOIN measurements m ON p.id = m.profile_id
GROUP BY p.id, f.platform_number, p.cycle_number, p.latitude, p.longitude, 
         p.measurement_date, p.measurement_year, p.measurement_season, p.data_mode;

COMMENT ON DATABASE floatchat IS 'ARGO Oceanographic Data Database optimized for LLM queries and conversational AI applications';