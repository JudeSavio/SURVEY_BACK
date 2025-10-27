# 🛠️ ChatSurvey Backend

This repository contains the backend for **ChatSurvey** — the API service that manages users, surveys, and survey state for the chat-like survey frontend.

---

## 🧩 Quick overview

The backend is a Python service. For local development follow the steps below to create a reproducible environment, install dependencies, configure environment variables, and run the server.

---

## 🔢 Required versions

- **Python:** `3.11.6` (use **pyenv** to install/manage)
- Recommended to use a virtual environment for dependency isolation.

---

## ⚙️ Setup — step by step

1. **Install Python 3.11.6 with `pyenv`**
```bash
pyenv install 3.11.6
pyenv local 3.11.6

### 2. Create a Virtual Environment
```bash
python3 -m venv venv

### 3. Activate the virtual environment

macOS / Linux (bash / zsh):

bash
Copy code
source venv/bin/activate

Windows (PowerShell):

powershell
Copy code
.\venv\Scripts\Activate.ps1
Install required packages


bash
Copy code
pip install -r requirements.txt
Create your .env

Copy .env.example to .env and fill in values:

bash
Copy code
cp .env.example .env
# then edit .env with your editor
(Refer to the team/email instructions for any organization-specific values or secrets.)

Add GROQ API key

Generate an API key for Groq and set it in your .env as:

ini
Copy code
GROQ_API_KEY=sk-<your_groq_api_key_here>
MongoDB (optional: use your own cluster)

If you plan to use your own MongoDB cluster, add the connection URI to .env:

php-template
Copy code
MONGO_URI=mongodb+srv://<username>:<password>@<cluster-url>/<dbname>?retryWrites=true&w=majority
Create the following collections in the database you point to:

users

surveys

user_survey_state

Sample document structure — users collection

json
Copy code
{
  "_id": "user123",
  "name": "Test User",
  "email": "test@example.com",
  "assignedSurveys": []
}

Note: _id can be any unique identifier string (or an ObjectId if you prefer). The example shows a simple string id.

Run the server

bash
Copy code
python main.py