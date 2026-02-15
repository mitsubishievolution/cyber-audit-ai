package com.cyberaudit.backend.repository;

import com.cyberaudit.backend.entity.Scan;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * ScanRepository - Database access interface for Scan entities
 * This interface provides methods to interact with the "scans" table.
 * 
 * Simple Explanation:
 * - You write method names following Spring conventions
 * - Spring automatically creates the SQL queries
 * 
 * How it works:
 * 1. Extend JpaRepository<Scan, Long>
 * - Scan: The entity we're working with
 * - Long: The type of the ID field (primary key)
 * 2. Spring provides: save(), findById(), findAll(), delete(), etc.
 * 3. Add custom methods using naming conventions
 */
@Repository // Marks this as a repository component so that spring will control it
public interface ScanRepository extends JpaRepository<Scan, Long> {

    // CUSTOM QUERY METHODS
    // Naming convention: findBy + FieldName + Condition

    /**
     * Find all scans with a specific status
     * 
     * Example usage:
     * List<Scan> completedScans = scanRepository.findByStatus("COMPLETED");
     * 
     * @param status The status to search for (e.g., "NEW", "COMPLETED")
     * @return List of scans with that status
     */
    List<Scan> findByStatus(String status);

    /**
     * Find all scans of a specific type
     */
    List<Scan> findByScanType(String scanType);

    /**
     * Find scans by URL (partial match)
     */
    List<Scan> findByTargetUrlContaining(String targetUrl);

    /**
     * Find a scan by exact URL
     */
    Optional<Scan> findByTargetUrl(String targetUrl);

    /**
     * Find scans created after a specific date/time
     * Useful for getting recent scans
     */
    List<Scan> findByCreatedAtAfter(LocalDateTime date);

    /**
     * Find scans created between two dates
     * Useful for reports and analytics
     * 
     */
    List<Scan> findByCreatedAtBetween(LocalDateTime start, LocalDateTime end);

    /**
     * Find scans by status and type (combined conditions)
     */
    List<Scan> findByStatusAndScanType(String status, String scanType);

    /**
     * Find scans ordered by creation date (newest first)
     * Great for showing recent activity
     */
    List<Scan> findAllByOrderByCreatedAtDesc();

    /**
     * Count scans by status
     * Useful for dashboard statistics
     */
    long countByStatus(String status);
}
