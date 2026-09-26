-- =======================================================
-- Parking System Database Initialization Script
-- =======================================================

-- 1. Create the Database if it does not already exist
CREATE DATABASE IF NOT EXISTS parking_system;

-- 2. Select the Database
USE parking_system;

-- 3. Create Table: previous_bookings
-- Stores vehicles that already possess an approved booking/parking allocation
CREATE TABLE IF NOT EXISTS previous_bookings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vehicle_number VARCHAR(20) UNIQUE NOT NULL,
    parking_slot VARCHAR(20),
    booking_status BOOLEAN DEFAULT TRUE
);

-- 4. Create Table: incoming_vehicles
-- Stores vehicles detected and logged by the live/video processing system
CREATE TABLE IF NOT EXISTS incoming_vehicles (
    series_number INT AUTO_INCREMENT PRIMARY KEY,
    vehicle_number VARCHAR(20) NOT NULL,
    parking_allocation VARCHAR(3) NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Insert Sample Data into previous_bookings
-- Realistic sample Indian registration numbers with allocated slots
INSERT INTO previous_bookings (vehicle_number, parking_slot, booking_status) VALUES
('MH12AB1234', 'A-12', TRUE),
('MH14CD5678', 'B-04', TRUE),
('MH12XY9999', 'C-21', TRUE),
('MH15PQ4567', 'A-05', TRUE),
('MH13EF7890', 'D-18', TRUE),
('DL01AB9876', 'E-03', TRUE),
('KA05MN3456', 'B-10', TRUE),
('HR26DK8392', 'C-07', TRUE)
ON DUPLICATE KEY UPDATE 
    parking_slot = VALUES(parking_slot),
    booking_status = VALUES(booking_status);

-- Note: incoming_vehicles table is intentionally left empty.
-- It will be populated dynamically by the detection engine.
