"""
test_plain_english.py

This is the terminal test script for the plain-English reporting layer.

Its purpose is to let you manually enter a vulnerability description,
send it to the running Flask API, and see the full output including:
- The ML model's predicted risk score
- The severity level (LOW, MEDIUM, HIGH, CRITICAL)
- The plain-English explanation (title, what this means, why it matters)
- The readability scores (Flesch Reading Ease and Grade Level)

How to use:
1. Make sure the virtual environment is active
2. Make sure the Flask API is running (python app.py in another terminal)
3. Run this script: python test_plain_english.py
4. Type in a vulnerability description when prompted
5. See the results printed in the terminal
"""

import requests  # Used to send HTTP requests to the Flask API
import json      # Used to format the output nicely


# The URL of the Flask API running locally
API_URL = "http://127.0.0.1:5000"


def print_separator():
    """Print a simple divider line to make the output easier to read."""
    print("\n" + "=" * 55 + "\n")


def print_result(result):
    """
    Display the API response in a readable format in the terminal.

    Args:
        result (dict): The JSON response from the /predict endpoint
    """
    print_separator()

    # Check if the API returned an error
    if "error" in result and result.get("error") is True:
        print("ERROR: Could not generate explanation.")
        print(f"Reason: {result.get('message', 'Unknown error')}")
        print_separator()
        return

    # Display the ML model's risk score and severity
    risk_score = result.get("risk_score", "N/A")
    severity = result.get("severity", "N/A")

    print(f"RISK SCORE:  {risk_score} / 10")
    print(f"SEVERITY:    {severity}")

    # Display the severity context (extra sentence about what severity means)
    severity_context = result.get("severity_context", "")
    if severity_context:
        print(f"\n{severity_context}")

    # Display the plain-English explanation
    plain = result.get("plain_english", {})
    if plain:
        print(f"\n--- Plain-English Explanation ---\n")
        print(f"Title:\n  {plain.get('title', 'N/A')}\n")
        print(f"What this means:\n  {plain.get('what_this_means', 'N/A')}\n")
        print(f"Why it matters:\n  {plain.get('why_it_matters', 'N/A')}")

    # Display the readability scores with pass/fail indicators
    readability = result.get("readability", {})
    if readability:
        flesch = readability.get("flesch_reading_ease", "N/A")
        grade = readability.get("flesch_kincaid_grade", "N/A")
        flesch_pass = readability.get("flesch_pass", False)
        grade_pass = readability.get("grade_pass", False)

        print(f"\n--- Readability Scores ---\n")
        print(f"  Flesch Reading Ease: {flesch}  {'✅ PASS' if flesch_pass else '❌ FAIL'} (target: 60+)")
        print(f"  Grade Level:         {grade}   {'✅ PASS' if grade_pass else '❌ FAIL'} (target: 10 or below)")

    print_separator()


def check_api_health():
    """
    Check that the Flask API is running before starting the test session.

    Returns:
        bool: True if API is reachable, False otherwise
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is running. Model status: {data.get('model_status', 'unknown')}")
            return True
        else:
            print(f"❌ API responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to the API.")
        print("   Make sure you have run: python app.py in another terminal.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error when checking API: {e}")
        return False


def send_vulnerability(description, vulnerability_type="", cvss=5.0):
    """
    Send a vulnerability to the API and return the response.

    Args:
        description (str): Text description of the vulnerability
        vulnerability_type (str): Optional category label (e.g. "SQL Injection")
        cvss (float): CVSS base score to use (0.0 to 10.0)

    Returns:
        dict or None: Parsed JSON response, or None if request failed
    """
    payload = {
        "description": description,
        "type": vulnerability_type,
        "cvss": cvss
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=15
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ API returned error status: {response.status_code}")
            print(f"   Response: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print("❌ Lost connection to the API mid-request.")
        return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None


def get_cvss_from_user():
    """
    Ask the user to enter a CVSS score and validate the input.

    Returns:
        float: A valid CVSS score between 0.0 and 10.0
    """
    while True:
        raw = input("Enter CVSS score (0.0 to 10.0, or press Enter to use 5.0): ").strip()

        # Use default if nothing entered
        if raw == "":
            return 5.0

        try:
            score = float(raw)
            if 0.0 <= score <= 10.0:
                return score
            else:
                print("   Please enter a number between 0.0 and 10.0.")
        except ValueError:
            print("   That is not a valid number. Please try again.")


def main():
    """
    Main loop that runs the interactive terminal test session.

    Checks the API is running, then repeatedly asks the user to enter
    a vulnerability description and displays the plain-English result.
    """
    print_separator()
    print("  CYBERSECURITY AUDIT - PLAIN-ENGLISH REPORTING TEST")
    print("  University Dissertation Project")
    print_separator()

    # Check API is available before starting
    print("Checking API connection...\n")
    if not check_api_health():
        print("\nPlease start the API first and then run this script again.")
        return

    print("\nType a vulnerability description to see the plain-English explanation.")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    # Run until the user types quit or presses Ctrl+C
    while True:
        try:
            print("-" * 55)

            # Get vulnerability description from user
            description = input("Vulnerability description: ").strip()

            # Exit if user types quit
            if description.lower() in ("quit", "exit", "q"):
                print("\nExiting test session. Goodbye!")
                break

            # Skip empty input
            if not description:
                print("Please enter a description to continue.\n")
                continue

            # Optionally ask for type and CVSS score
            vuln_type = input("Vulnerability type (optional, press Enter to skip): ").strip()
            cvss = get_cvss_from_user()

            print("\nSending to API...")

            # Send to API and display result
            result = send_vulnerability(description, vuln_type, cvss)

            if result:
                print_result(result)
            else:
                print("No result returned. Check the API output for details.\n")

        except KeyboardInterrupt:
            print("\n\nExiting test session. Goodbye!")
            break


if __name__ == "__main__":
    main()
