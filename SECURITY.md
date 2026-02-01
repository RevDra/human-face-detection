# Security Policy

## 📦 Supported Versions

Only the latest stable release is currently supported with security updates.

| Version | Supported          | Notes |
| :-----: | :----------------: | :---- |
| **1.0.x** | :white_check_mark: | Current Stable Release (YOLOv12) |
| < 1.0   | :x:                | Old development versions |

## 🐞 Reporting a Vulnerability

We take the security of this project seriously. If you find a vulnerability, please follow these steps:

### How to Report
**DO NOT** open a public issue on GitHub for sensitive security vulnerabilities. Publicly disclosing a vulnerability can put the entire community at risk.

Instead, please send an email to: **[YOUR_EMAIL]@gmail.com**

Please include:
* Description of the vulnerability.
* Steps to reproduce the issue.
* Potential impact.

### What to Expect
* I will acknowledge your email within **48 hours**.
* I will verify the issue and keep you updated on the progress.
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
