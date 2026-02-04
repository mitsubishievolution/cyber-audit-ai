-- DATABASE SCHEMA
-- This file creates the database tables and adds sample data
-- It runs automatically when the application starts (see application-dev.properties)

-- TABLE 1: SCANS
-- This table stores information about each security scan performed
CREATE TABLE scans (

    id SERIAL PRIMARY KEY,
    target_url TEXT NOT NULL, -- must have a value
    -- Current status of the scan
    -- Possible values: 'NEW', 'IN_PROGRESS', 'COMPLETED', 'FAILED'
    -- DEFAULT 'NEW' means new scans start with this status
    status VARCHAR(50) DEFAULT 'NEW',
    -- Type of scan performed
    -- Possible values: 'QUICK', 'FULL'
    scan_type VARCHAR(20) DEFAULT 'FULL',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, --Time stamp of scan
    -- When the scan finished (NULL if still running)
    completed_at TIMESTAMP NULL
);


-- TABLE 2: VULNERABILITIES
-- This table stores security vulnerabilities found during scans
-- Each row represents one vulnerability
-- Multiple vulnerabilities can belong to one scan

CREATE TABLE vulnerabilities (
    
    id SERIAL PRIMARY KEY,
    
    -- Foreign key: Links to the scans table
    -- REFERENCES scans(id) creates a relationship between tables
    scan_id INTEGER REFERENCES scans(id),
    
    -- Name of the vulnerability
    name TEXT NOT NULL,
    
    -- How serious the vulnerability is
    -- Possible values: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    severity VARCHAR(20) NOT NULL,
    
    -- Explanation of what was found
    description TEXT,
    
    -- Which tool found this vulnerability
    source_tool VARCHAR(20),
    
    -- AI-predicted risk score (0.0 to 10.0)
    risk_score DECIMAL(3,1),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- SAMPLE DATA (For Demonstration)
-- This pre-populates the database with example data

-- Add 2 example scans
INSERT INTO scans (target_url, status, scan_type, completed_at) VALUES 
    -- Scan 1: Completed scan of example.com
    ('https://example.com', 'COMPLETED', 'FULL', CURRENT_TIMESTAMP),
    
    -- Scan 2: In-progress scan of test.com
    ('https://test.com', 'IN_PROGRESS', 'QUICK', NULL);

-- Add 4 example vulnerabilities
-- First two belong to scan 1, last two belong to scan 2
INSERT INTO vulnerabilities (scan_id, name, severity, description, source_tool, risk_score) VALUES 
    -- Vulnerability 1: SQL Injection found by ZAP
    (1, 'SQL Injection', 'HIGH', 'Potential SQL injection vulnerability found in login form. Attackers could access database.', 'ZAP', 8.5),
    
    -- Vulnerability 2: Open HTTP port found by NMAP
    (1, 'Open Port 80 (HTTP)', 'MEDIUM', 'HTTP port is open and accepting connections. Should use HTTPS instead.', 'NMAP', 5.0),
    
    -- Vulnerability 3: XSS found by ZAP
    (1, 'Cross-Site Scripting (XSS)', 'HIGH', 'Unvalidated user input allows JavaScript injection in search field.', 'ZAP', 7.8),
    
    -- Vulnerability 4: Open SSH port found by NMAP
    (2, 'Open Port 22 (SSH)', 'LOW', 'SSH port is accessible. Ensure strong authentication is enabled.', 'NMAP', 3.2);

-- ====================================
-- VERIFICATION QUERIES (For Testing)
-- ====================================
-- You can run these queries in H2 Console to verify the data:
--
-- 1. View all scans:
--    SELECT * FROM scans;
--
-- 2. View all vulnerabilities:
--    SELECT * FROM vulnerabilities;
--
-- 3. View vulnerabilities with their scan details:
--    SELECT s.target_url, s.status, v.name, v.severity, v.risk_score 
--    FROM scans s 
--    LEFT JOIN vulnerabilities v ON s.id = v.scan_id;
--
-- 4. Count vulnerabilities by severity:
--    SELECT severity, COUNT(*) as count 
--    FROM vulnerabilities 
--    GROUP BY severity 
--    ORDER BY count DESC;
