# Contributing to YOLOv12 Face Detection

First off, thanks for taking the time to contribute! 🎉

We welcome contributions from everyone. By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🛠️ Development Workflow

Since this repository uses **Strict Rulesets** and **CI/CD Pipelines**, you cannot push directly to the `main` branch. Please follow this workflow:

### 1. Fork & Clone
Fork the repository to your GitHub account, then clone it locally:
```bash
git clone [https://github.com/YOUR_USERNAME/Human_face_detection.git](https://github.com/YOUR_USERNAME/Human_face_detection.git)
cd Human_face_detection
```

### 2. Environment Setup
We recommend using a virtual environment:
```bash
# Create venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies (CPU optimized for dev)
pip install -r requirements.txt
```
**Note**: You need to download the models manually for local development. Run:
```bash
# Linux/Mac
bash config/deploy.sh check

# Windows
config\deploy.bat check
```

### 3. Create a Branch
Never work on the `main` branch. Create a feature branch:
```bash
git checkout -b feature/amazing-feature
# or
git checkout -b fix/bug-fix-name
```

### 4. Coding Standards
- Python: Follow PEP 8 guidelines.
- Structure: Keep source code in `src/` and configs in `config/`.
- Requirements: If you add a new library, update `requirements.txt`, but avoid upgrading `numpy` to 2.0+ (Keep it `<2.0.0`).

### 5. Local Testing
Before submitting, ensure the app runs locally, and Docker builds successfully:
**Test App:**
```bash
python src/web_app.py
```
**Test Docker Build (Crucial):**
```bash
docker build -f config/Dockerfile -t test-build .
```

### 6. Commit Messages
We encourage Semantic Commit Messages to keep the history clean:
- `feat: add new video processing logic`
- `fix: resolve crash on upload`
- `docs: update deployment guide`
- `style: format code with black`
- `chore: update dockerfile`

## 🚀 Submitting a Pull Request (PR)
1. Push your branch to your fork:
  ```bash
  git push origin feature/amazing-feature
  ```
2. Open a **Pull Request** to the `main` branch of this repository.
3. **Wait for Checks**: The **Docker** GitHub Action will run automatically.
4. **Green Check?** If the build passes, your code is safe to merge.
5. **Red Cross?** If the build fails, fix the errors and push again.

## 🐛 Reporting Bugs
If you find a bug, please create an issue with:
- Steps to reproduce.
- Expected vs. Actual behavior.
- Screenshots (if applicable).

---
Happy Coding! 🚀
