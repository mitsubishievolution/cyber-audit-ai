package com.cyberaudit.backend.entity;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * - When a user submits a URL to scan, we create a Scan object
 * - This object stores: URL, status, when it started, when it finished
 */
@Entity // represents a table in the database
@Table(name = "scans") // table being used
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Scan {
    // PRIMARY KEY
    @Id // Marks this field as the primary key
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // SCAN DETAILS
    /**
     * The website URL being scanned
     * nullable = false means this field is required (cannot be empty)
     */
    @Column(nullable = false)
    private String targetUrl;

    /**
     * Current status of the scan
     * Possible values: "NEW", "IN_PROGRESS", "COMPLETED", "FAILED"
     */
    private String status;

    /**
     * Type of scan being performed
     */
    private String scanType;

    // TIMESTAMPS
    /**
     * When the scan was created/started
     * 
     * @CreationTimestamp automatically sets this to current time
     */
    @CreationTimestamp
    @Column(updatable = false) // Once set, this field cannot be changed
    private LocalDateTime createdAt;

    /**
     * When the scan completed (null if still running)
     * If this is NULL, the scan is still in progress
     * If this has a value, the scan has finished
     */
    private LocalDateTime completedAt;

    // RELATIONSHIP WITH VULNERABILITIES
    /**
     * List of vulnerabilities found during this scan
     * 
     * cascade = CascadeType.ALL: If scan is deleted, delete its vulnerabilities too
     * orphanRemoval = true: If vulnerability is removed from list, delete it from
     * database
     */

    @OneToMany(mappedBy = "scan", // Links to the "scan" field in Vulnerability class
            cascade = CascadeType.ALL, // Operations on Scan affect Vulnerabilities
            orphanRemoval = true, // Remove vulnerabilities if they're removed from this list
            fetch = FetchType.LAZY // Only load vulnerabilities when accessed (performance optimization)
    )
    private List<Vulnerability> vulnerabilities = new ArrayList<>();

    // CUSTOM CONSTRUCTORS
    /**
     * Simple constructor for creating new scans
     * Used when a user submits a URL to scan
     * 
     * @param targetUrl
     * @param status
     * @param scanType
     */
    public Scan(String targetUrl, String status, String scanType) {
        this.targetUrl = targetUrl;
        this.status = status;
        this.scanType = scanType;
    }

    // HELPER METHODS
    /**
     * Add a vulnerability to this scan
     * Helper method to maintain the relationship properly
     * 
     * @param vulnerability The vulnerability to add
     */
    public void addVulnerability(Vulnerability vulnerability) {
        // Add to our list
        vulnerabilities.add(vulnerability);
        vulnerability.setScan(this);
    }

    /**
     * Remove a vulnerability from this scan
     * 
     * @param vulnerability The vulnerability to remove
     */
    public void removeVulnerability(Vulnerability vulnerability) {
        // Remove from our list
        vulnerabilities.remove(vulnerability);

        // Clear the reverse relationship
        vulnerability.setScan(null);
    }

    /**
     * Check if scan is completed
     * 
     * @return true if scan has finished, false otherwise
     */
    public boolean isCompleted() {
        return completedAt != null;
    }

    /**
     * Calculate how long the scan took (in seconds)
     * Only works if scan is completed
     * 
     * @return Duration in seconds, or 0 if not completed
     */
    public long getDurationInSeconds() {
        if (completedAt == null || createdAt == null) {
            return 0;
        }
        return java.time.Duration.between(createdAt, completedAt).getSeconds();
    }

    /**
     * Override toString to prevent infinite loop with Vulnerability relationship
     * 
     * @return String representation without the vulnerabilities list
     */
    @Override
    public String toString() {
        return "Scan{" +
                "id=" + id +
                ", targetUrl='" + targetUrl + '\'' +
                ", status='" + status + '\'' +
                ", scanType='" + scanType + '\'' +
                ", createdAt=" + createdAt +
                ", completedAt=" + completedAt +
                ", vulnerabilityCount=" + (vulnerabilities != null ? vulnerabilities.size() : 0) +
                '}';
    }
}
