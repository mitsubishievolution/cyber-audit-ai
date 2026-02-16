package com.cyberaudit.backend.service;

import com.cyberaudit.backend.entity.Vulnerability;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * NmapService - Scans websites for open network ports
 * 
 * Uses NMAP (Network Mapper) to scan target websites and identify open ports.
 * Open ports can be security vulnerabilities if not properly secured.
 * 
 */
@Service
public class NmapService {

    private final VulnerabilityService vulnerabilityService;

    // Timeout for NMAP scans (60 seconds)
    private static final int SCAN_TIMEOUT_SECONDS = 60;

    public NmapService(VulnerabilityService vulnerabilityService) {
        this.vulnerabilityService = vulnerabilityService;
    }

    /**
     * Scan a target URL for open ports
     * 
     * @param targetUrl The website URL to scan
     * @param scanId    The scan ID to associate vulnerabilities with
     */
    public void scan(String targetUrl, Long scanId) {
        try {
            // Step 1: Extract domain from URL (e.g., "https://example.com" → "example.com")
            String domain = extractDomain(targetUrl);

            // Step 2: Run NMAP command
            String nmapOutput = runNmapCommand(domain);

            // Step 3: Parse results and save vulnerabilities
            parseAndSaveVulnerabilities(nmapOutput, scanId);

        } catch (Exception e) {
            // If anything goes wrong, save an error vulnerability
            saveErrorVulnerability(scanId, e.getMessage());
        }
    }

    /**
     * Extract domain name from full URL
     * Converts "https://example.com/path" to "example.com"
     */
    private String extractDomain(String urlString) throws Exception {
        try {
            URL url = new URL(urlString);
            return url.getHost(); // Returns just the domain part
        } catch (Exception e) {
            throw new Exception("INVALID_URL: Could not simplify URL");
        }
    }

    /**
     * Execute NMAP command and return output
     */
    private String runNmapCommand(String domain) throws Exception {
        try {
            // Build the NMAP command
            ProcessBuilder processBuilder = new ProcessBuilder(
                    "nmap",
                    "-sV", // Service version detection
                    "-T4", // Faster scan timing
                    "-Pn", // Don't ping first
                    "--top-ports", "20", // Scan top 20 most common ports
                    domain);

            // Start the process
            Process process = processBuilder.start();

            // Wait for completion with timeout
            boolean finished = process.waitFor(SCAN_TIMEOUT_SECONDS, TimeUnit.SECONDS);

            if (!finished) {
                // Timeout occurred
                process.destroy();
                throw new Exception("TIMEOUT: Scan timed out after " + SCAN_TIMEOUT_SECONDS + " seconds");
            }

            // Check if NMAP command was successful
            if (process.exitValue() != 0) {
                throw new Exception("NMAP_ERROR: NMAP command failed");
            }

            // Read the output
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()));

            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            return output.toString();

        } catch (java.io.IOException e) {
            throw new Exception("NMAP_NOT_FOUND: NMAP is not installed on this system");
        } catch (InterruptedException e) {
            throw new Exception("SCAN_INTERRUPTED: Scan was interrupted");
        }
    }

    /**
     * Parse NMAP output and save vulnerabilities
     * 
     * NMAP output looks like:
     * PORT STATE SERVICE
     * 80/tcp open http
     * 443/tcp open https
     */
    private void parseAndSaveVulnerabilities(String nmapOutput, Long scanId) {
        List<Vulnerability> vulnerabilities = new ArrayList<>();
        String[] lines = nmapOutput.split("\n");

        boolean foundPorts = false;

        // Look for lines with port information
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i];
            line = line.trim();

            // Skip header lines and empty lines
            if (line.isEmpty() || line.startsWith("PORT") || line.startsWith("Nmap")) {
                continue;
            }

            // Check if line contains port info (e.g., "80/tcp open http")
            if (line.matches("\\d+/tcp\\s+open.*")) {
                foundPorts = true;

                // Extract port number (first part before "/")
                String portStr = line.split("/")[0].trim();
                int port = Integer.parseInt(portStr);

                // Extract service name (last part)
                String[] parts = line.split("\\s+");
                String service = parts.length > 2 ? parts[2] : "unknown";

                // Create vulnerability for this open port
                Vulnerability vuln = createPortVulnerability(scanId, port, service);
                if (vuln != null) {
                    vulnerabilities.add(vuln);
                }
            }
        }

        // Save all found vulnerabilities
        for (Vulnerability vuln : vulnerabilities) {
            vulnerabilityService.addVulnerability(
                    scanId,
                    vuln.getName(),
                    vuln.getSeverity(),
                    vuln.getDescription(),
                    "NMAP");
        }

        // If no ports found, note that
        if (!foundPorts) {
            vulnerabilityService.addVulnerability(
                    scanId,
                    "No Open Ports Detected",
                    "LOW",
                    "NMAP scan completed but found no open ports. This is generally good!",
                    "NMAP");
        }
    }

    /**
     * Create a vulnerability for an open port
     * Maps port numbers to security issues
     */
    private Vulnerability createPortVulnerability(Long scanId, int port, String service) {
        String name;
        String severity;
        String description;

        // Map common ports to vulnerabilities
        switch (port) {
            case 80:
                name = "Open Port 80 (HTTP)";
                severity = "MEDIUM";
                description = "Unencrypted web traffic detected on port 80. "
                        + "Data sent over HTTP can be intercepted.";
                break;

            case 443:
                name = "Open Port 443 (HTTPS)";
                severity = "LOW";
                description = "Secure HTTPS connection detected on port 443. " + "This is expected for web servers.";
                break;

            case 22:
                name = "Open Port 22 (SSH)";
                severity = "MEDIUM";
                description = "SSH remote access is available on port 22. "
                        + "Ensure strong authentication is configured.";
                break;

            case 21:
                name = "Open Port 21 (FTP)";
                severity = "HIGH";
                description = "FTP file transfer service detected on port 21. " + "FTP transmits data in plain text.";
                break;

            case 23:
                name = "Open Port 23 (Telnet)";
                severity = "CRITICAL";
                description = "Telnet service detected on port 23. " + "Telnet is unencrypted and highly insecure.";
                break;

            case 8080:
                name = "Open Port 8080 (HTTP-Alt)";
                severity = "LOW";
                description = "Alternative HTTP port detected on 8080. "
                        + "Commonly used for development or proxy servers.";
                break;

            case 8443:
                name = "Open Port 8443 (HTTPS-Alt)";
                severity = "LOW";
                description = "Alternative HTTPS port detected on 8443. "
                        + "Commonly used for administrative interfaces.";
                break;

            default:
                // Generic open port
                name = "Open Port " + port + " (" + service + ")";
                severity = "MEDIUM";
                description = "Port " + port + " is open and running " + service + " service. "
                        + "Review if this port should be publicly accessible.";
                break;
        }

        // Create a vulnerability object (not saved yet)
        Vulnerability vuln = new Vulnerability();
        vuln.setName(name);
        vuln.setSeverity(severity);
        vuln.setDescription(description);

        return vuln;
    }

    /**
     * Save an error as a vulnerability
     * Called when scan fails
     */
    private void saveErrorVulnerability(Long scanId, String errorMessage) {
        vulnerabilityService.addVulnerability(
                scanId,
                "NMAP Scan Error",
                "LOW",
                "NMAP scan encountered an error: " + errorMessage,
                "NMAP");
    }
}
