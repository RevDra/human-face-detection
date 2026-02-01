# Security Policy

## 📦 Supported Versions

Only the latest stable release is currently supported with security updates.

| Version | Supported          | Notes |
| :-----: | :----------------: | :---- |
| **1.1.x** | :white_check_mark: | **Current Stable & Secure** (Docker Support) |
| 1.0.x   | :x:                | Vulnerable (Please upgrade immediately) |

## 🐞 Reporting a Vulnerability

We take the security of this project seriously. If you find a vulnerability, please follow the guidelines below.

### 🚫 DO NOT open Public Issues
**Never** report security vulnerabilities via public GitHub issues. Publicly disclosing a vulnerability can put the entire community at risk before a fix is available.

### ✅ Preferred Method (Private Reporting)
We strongly encourage you to use GitHub's **Private Vulnerability Reporting** feature. This allows us to collaborate on a fix in a secure, private environment.

1. Go to the **Security** tab of this repository.
2. Click on the **Report a vulnerability** button.
3. Fill in the details (Description, Impact, Steps to reproduce).

### 📧 Alternative Method (Email)
If you are unable to use the GitHub reporting tool, you can send an email to: **luongminhngoc0@gmail.com**

Please include:
* Description of the vulnerability.
* Steps to reproduce the issue.
* Potential impact.

### What to Expect
* We will acknowledge your report within **48 hours**.
* We will verify the issue and keep you updated on the progress.
* Once fixed, a patch will be released, and you will be credited (if you wish) in the release notes.

## 🔒 Security Scope

### In Scope
* Vulnerabilities in the Flask Web Application code (`src/web_app.py`).
* Issues with the Docker configuration (`Dockerfile`, `docker-compose.yml`).
* Cross-Site Scripting (XSS) or File Upload vulnerabilities in the web interface.

### Out of Scope
* Vulnerabilities in third-party libraries (e.g., `ultralytics`, `pytorch`, `flask`). Please report those to the respective maintainers.
* Attacks requiring physical access to the user's device.
* Spam or social engineering attacks.

## ⚠️ Important Note for Users
This application is designed for **Research and Educational purposes**.
* **Do not** deploy this application to the public internet without adding an Authentication layer (e.g., Nginx Basic Auth, OAuth).
* The default configuration runs in "Development Mode" or basic Gunicorn mode without built-in user management.
