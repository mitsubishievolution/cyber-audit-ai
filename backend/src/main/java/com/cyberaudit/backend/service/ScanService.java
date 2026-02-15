package com.cyberaudit.backend.service;

import com.cyberaudit.backend.entity.Scan;
import com.cyberaudit.backend.repository.ScanRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * ScanService - Handles all scan-related operations
 */
@Service // Spring magic: manages this class automatically
public class ScanService {

    // The tool that talks to the database
    private final ScanRepository scanRepository;

    // Spring calls this when starting up
    public ScanService(ScanRepository scanRepository) {
        this.scanRepository = scanRepository;
    }

    // CREATE - Making new scans
    /**
     * Create a new scan with a specific type
     * Uses this when a user submits a URL to scan
     */
    public Scan createScan(String targetUrl, String scanType) {
        // Make a new scan object
        Scan scan = new Scan();
        scan.setTargetUrl(targetUrl);
        scan.setStatus("NEW"); // All scans start as "NEW"
        scan.setScanType(scanType);

        // Save it and return (database will give it an ID)
        return scanRepository.save(scan);
    }

    /**
     * Create a new scan (defaults to FULL scan type)
     */
    public Scan createScan(String targetUrl) {
        return createScan(targetUrl, "FULL");
    }

    // READ - Finding scans
    /**
     * Find a scan by its ID
     * Returns are optional because the scan might not exist
     */
    public Optional<Scan> getScanById(Long id) {
        return scanRepository.findById(id);
    }

    /**
     * Get all scans in the system
     */
    public List<Scan> getAllScans() {
        return scanRepository.findAll();
    }

    /**
     * Get scans by their status
     * Example: Find all "COMPLETED" scans
     */
    public List<Scan> getScansByStatus(String status) {
        return scanRepository.findByStatus(status);
    }

    /**
     * Search for scans by URL
     * Finds scans that match part of the URL
     */
    public List<Scan> getScansByUrl(String targetUrl) {
        return scanRepository.findByTargetUrlContaining(targetUrl);
    }

    /**
     * Get recent scans (newest first)
     * Good for showing latest activity
     */
    public List<Scan> getRecentScans() {
        return scanRepository.findAllByOrderByCreatedAtDesc();
    }

    // UPDATE - Changing scan information
    /**
     * Update a scan's status
     * Use this to track progress: NEW → IN_PROGRESS → COMPLETED
     */
    public Scan updateScanStatus(Long id, String newStatus) {
        // Try to find the scan
        Optional<Scan> scanOptional = scanRepository.findById(id);

        if (scanOptional.isPresent()) {
            Scan scan = scanOptional.get();
            scan.setStatus(newStatus);

            // If marking as done, record the completion time
            if ("COMPLETED".equals(newStatus) || "FAILED".equals(newStatus)) {
                scan.setCompletedAt(LocalDateTime.now());
            }

            return scanRepository.save(scan);
        }

        return null; // Scan not found
    }

    /**
     * Mark scan as currently running
     * Easier than calling updateScanStatus()
     */
    public Scan markScanInProgress(Long id) {
        return updateScanStatus(id, "IN_PROGRESS");
    }

    /**
     * Mark scan as finished successfully
     */
    public Scan markScanCompleted(Long id) {
        return updateScanStatus(id, "COMPLETED");
    }

    /**
     * Mark scan as failed (something went wrong)
     */
    public Scan markScanFailed(Long id) {
        return updateScanStatus(id, "FAILED");
    }

    // DELETE - Removing scans
    /**
     * Delete a scan
     * This will also delete all the vulnerabilities found in that scan
     */
    public void deleteScan(Long id) {
        scanRepository.deleteById(id);
    }

    // Helpful extras
    /**
     * Count how many scans exist
     */
    public long countAllScans() {
        return scanRepository.count();
    }

    /**
     * Count scans with a specific status
     */
    public long countScansByStatus(String status) {
        return scanRepository.countByStatus(status);
    }

    /**
     * Check if a scan exists
     */
    public boolean scanExists(Long id) {
        return scanRepository.existsById(id);
    }
}
