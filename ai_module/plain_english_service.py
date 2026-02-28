"""

This file is responsible for generating plain-English explanations of
cybersecurity vulnerabilities. Its job is to take raw technical findings from the ML model and turn them into simple, clear text that a non-technical business owner can actually understand and act on.

How it works:
1. A vulnerability is passed in (e.g. "SQL injection in login page")
2. The service maps it to a category (e.g. SQL_INJECTION)
3. It looks up the matching template from vulnerability_templates.json
4. It calculates readability scores (Flesch Reading Ease and Grade Level)
5. It returns a structured response with the plain-English explanation
"""

import json
import os
import textstat  # Library that calculates readability scores


class PlainEnglishService:
    """
    Generates plain-English vulnerability explanations from templates.

    This class loads a JSON file of pre-written templates and selects
    the right one based on what type of vulnerability was found.
    It also calculates readability scores to verify the text is simple
    enough for a non-technical audience.
    """

    def __init__(self, templates_path=None):
        """
        Load the vulnerability templates from the JSON file.

        Args:
            templates_path (str): Path to the templates JSON file.
            If not provided, it will look for the file relative to this script's location.
        """
        # If no path is given, build the path relative to this file's location
        if templates_path is None:
            # Get the directory this script is in (ai_module/)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to the project root, then into data/
            templates_path = os.path.join(
                os.path.dirname(current_dir),
                "data",
                "vulnerability_templates.json"
            )

        # Load the templates from the JSON file
        self.templates = self._load_templates(templates_path)

    def _load_templates(self, path):
        """
        Read and parse the templates JSON file.

        Args:
            path (str): Full path to the JSON templates file

        Returns:
            dict: Parsed template data

        Raises:
            FileNotFoundError: If the templates file does not exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Templates file not found at: {path}\n"
                "Make sure data/vulnerability_templates.json exists."
            )

        with open(path, 'r') as f:
            return json.load(f)

    def map_to_category(self, description, vulnerability_type=None):
        """
        Determine which template category best fits this vulnerability.

        This works by checking the description and type for known keywords.
        For example, if "SQL" appears in the type, we map it to SQL_INJECTION.
        If nothing matches, we fall back to the GENERIC category.

        Args:
            description (str): Text description of the vulnerability
            vulnerability_type (str): Optional type label (e.g. "SQL Injection")

        Returns:
            str: Category key that matches a template (e.g. "SQL_INJECTION")
        """
        # Normalise to lowercase so comparisons are case-insensitive
        desc = description.lower()
        vuln_type = (vulnerability_type or "").lower()

        # Check for SQL injection keywords
        if "sql" in vuln_type or "sql injection" in desc or "sql" in desc:
            return "SQL_INJECTION"

        # Check for Cross-Site Scripting (XSS)
        if "xss" in vuln_type or "cross-site scripting" in desc or "xss" in desc:
            return "XSS"

        # Check for SSH (usually port 22)
        if "ssh" in desc or "ssh" in vuln_type or "port 22" in desc:
            return "OPEN_PORT_SSH"

        # Check for FTP (usually port 21)
        if "ftp" in desc or "ftp" in vuln_type or "port 21" in desc:
            return "OPEN_PORT_FTP"

        # Check for HTTPS (port 443)
        if "https" in desc or "port 443" in desc or "ssl" in vuln_type:
            return "OPEN_PORT_HTTPS"

        # Check for HTTP (port 80)
        if "http" in desc or "port 80" in desc or "web server" in desc:
            return "OPEN_PORT_HTTP"

        # Check for any other open port references
        if "open port" in desc or "port" in desc:
            return "OPEN_PORT_GENERIC"

        # Check for outdated or old software versions
        if "outdated" in desc or "old version" in desc or "deprecated" in desc:
            return "OUTDATED_SOFTWARE"

        # Check for SSL/TLS encryption issues
        if "ssl" in desc or "tls" in desc or "certificate" in desc or "encryption" in desc:
            return "WEAK_SSL"

        # Check for default credentials
        if "default" in desc and ("password" in desc or "credential" in desc):
            return "DEFAULT_CREDENTIALS"

        # Check for information disclosure
        if "disclosure" in desc or "information leak" in desc or "sensitive data" in desc:
            return "INFO_DISCLOSURE"

        # Check for missing security headers
        if "header" in desc or "security header" in desc:
            return "MISSING_HEADERS"

        # If nothing matched, use the generic fallback template
        return "GENERIC"

    def get_severity_label(self, risk_score):
        """
        Convert a numeric CVSS risk score into a severity label.

        CVSS scoring follows an industry standard scale:
        - 9.0 to 10.0 = Critical
        - 7.0 to 8.9  = High
        - 4.0 to 6.9  = Medium
        - 0.1 to 3.9  = Low
        - 0.0         = None / Unknown

        Args:
            risk_score (float): ML predicted risk score (0.0 to 10.0)

        Returns:
            str: Severity label (CRITICAL, HIGH, MEDIUM, LOW, or UNKNOWN)
        """
        if risk_score is None:
            return "UNKNOWN"
        if risk_score >= 9.0:
            return "CRITICAL"
        if risk_score >= 7.0:
            return "HIGH"
        if risk_score >= 4.0:
            return "MEDIUM"
        if risk_score > 0.0:
            return "LOW"
        return "UNKNOWN"

    def calculate_readability(self, text):
        """
        Calculate readability scores for a piece of text.

        Uses the textstat library to compute two standard metrics:
        - Flesch Reading Ease: Higher score = easier to read (target: 60+)
        - Flesch-Kincaid Grade Level: Lower = simpler (target: 10 or below)

        Args:
            text (str): The text to analyse

        Returns:
            dict: Readability scores and whether they meet dissertation targets
        """
        flesch_score = textstat.flesch_reading_ease(text)
        grade_level = textstat.flesch_kincaid_grade(text)

        # Check against dissertation success thresholds
        flesch_pass = flesch_score >= 60.0   # Target: 60 or above
        grade_pass = grade_level <= 10.0     # Target: grade 10 or below

        return {
            "flesch_reading_ease": round(flesch_score, 1),
            "flesch_kincaid_grade": round(grade_level, 1),
            "flesch_pass": flesch_pass,   # True if meets dissertation target
            "grade_pass": grade_pass      # True if meets dissertation target
        }

    def generate_explanation(self, description, vulnerability_type=None, risk_score=None):
        """
        Generate a full plain-English explanation for a vulnerability.

        This is the main method called by the API. It ties everything together:
        1. Map vulnerability to a category
        2. Look up the matching template
        3. Get severity from risk score
        4. Calculate readability scores
        5. Return the full structured result

        Args:
            description (str): Text description of the vulnerability
            vulnerability_type (str): Optional type label (e.g. "SQL Injection")
            risk_score (float): ML predicted risk score (0.0 to 10.0)

        Returns:
            dict: Full explanation including plain-English text and readability scores
                  Returns an error dict if template lookup fails
        """
        try:
            # Step 1: Determine which category this vulnerability belongs to
            category = self.map_to_category(description, vulnerability_type)

            # Step 2: Look up the template for that category
            categories = self.templates.get("categories", {})
            template = categories.get(category)

            # If the category is not found, fall back to the GENERIC template
            if not template:
                template = categories.get("GENERIC", {})
                category = "GENERIC"

            # Step 3: Convert the risk score to a severity label
            severity = self.get_severity_label(risk_score)

            # Step 4: Get the severity context sentence
            severity_context = self.templates.get("severity_context", {}).get(
                severity, ""
            )

            # Step 5: Build the full text to analyse for readability
            full_text = (
                f"{template.get('title', '')}. "
                f"{template.get('what_this_means', '')} "
                f"{template.get('why_it_matters', '')}"
            )

            # Step 6: Calculate readability scores on the full explanation
            readability = self.calculate_readability(full_text)

            # Step 7: Return the complete structured response
            return {
                "category": category,
                "severity": severity,
                "severity_context": severity_context,
                "plain_english": {
                    "title": template.get("title", "Security Issue Found"),
                    "what_this_means": template.get("what_this_means", ""),
                    "why_it_matters": template.get("why_it_matters", "")
                },
                "readability": readability
            }

        except Exception as e:
            # If anything goes wrong, return a clear error rather than crashing
            return {
                "error": True,
                "message": f"Could not generate plain-English explanation: {str(e)}"
            }
