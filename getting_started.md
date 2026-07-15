# Getting Started: DXC RAG Diagnostics Application

This guide provides step-by-step instructions to set up, build, configure, and run the DXC RAG Diagnostics application on **Linux/macOS** and **Windows** systems.

---

## 📋 Prerequisites

Before starting, ensure you have Python installed:
* **Python**: Version `3.10` or higher (tested with `3.13.5`)
* **pip**: Python package installer (normally bundled with Python)

### 🪟 Windows Setup Note:
When installing Python on Windows:
1. Download the installer from [python.org](https://www.python.org/downloads/).
2. **CRITICAL**: In the installer window, check the box that says **"Add Python.exe to PATH"** before clicking install. Otherwise, the `python` command will not be recognized in your terminal.

---

## 🛠️ Step 1: Create a Virtual Environment

It is recommended to run the application within an isolated virtual environment to prevent package conflicts.

Open your terminal (Bash on Linux/macOS, or Command Prompt / PowerShell on Windows) and navigate to the project directory:

```bash
cd /path/to/hybrid-rag-diagnostics
```

### Create the environment:
* **Linux/macOS**:
  ```bash
  python3 -m venv .venv
  ```
* **Windows (CMD or PowerShell)**:
  ```powershell
  python -m venv .venv
  ```

### Activate the environment:
* **Linux/macOS**:
  ```bash
  source .venv/bin/activate
  ```
* **Windows (Command Prompt)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **Windows (PowerShell)**:
  > [!TIP]
  > If PowerShell blocks script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first in your session.
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

---

## 📦 Step 2: Install Dependencies

With the virtual environment activated, run the following command to install the RAG components:

```bash
pip install -r requirements.txt
```

> [!NOTE]
> This step installs Streamlit, LangChain, Chroma DB, Hugging Face Hub, PyPDF, and Ragas. The initial installation may take a few minutes as it downloads numerical and machine learning packages like Torch.

---

## 🔑 Step 3: Configure API Keys

To generate answers using LLMs, you need API keys. You can set them as environment variables in your terminal session before launching the app.

### 🐧 Linux/macOS (Bash):
```bash
export GROQ_API_KEY="your_groq_api_key_here"
export GEMINI_API_KEY="your_google_gemini_api_key_here"
export OPENAI_API_KEY="your_openai_api_key_here"
```

### 🪟 Windows (Command Prompt - CMD):
```cmd
set GROQ_API_KEY=your_groq_api_key_here
set GEMINI_API_KEY=your_google_gemini_api_key_here
set OPENAI_API_KEY=your_openai_api_key_here
```

### 🪟 Windows (PowerShell):
```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
$env:GEMINI_API_KEY="your_google_gemini_api_key_here"
$env:OPENAI_API_KEY="your_openai_api_key_here"
```

*Alternatively, you can type your keys directly into the sidebar text input inside the browser UI once the application is running.*

---

## 🚀 Step 4: Run the Application

Start the Streamlit dashboard by executing:

```bash
streamlit run app.py
```

Upon a successful startup, Streamlit will print the local access URLs:
* **Local URL**: `http://localhost:8501`
* **Network URL**: `http://192.168.x.x:8501`

Your default web browser should open the page automatically. If it does not, copy and paste the `Local URL` into your browser.

---

## 🧩 Step 5: Seeding the Database (Automatic)

On the very first launch, if the system detects that the database is empty:
1. It automatically downloads the embedding model (`BAAI/bge-base-en-v1.5`) and the Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to your local Hugging Face cache.
2. It automatically indexes the default document `HDFC-Surgicare-Plan-101N043V01.pdf` from the workspace root into Chroma DB and creates the BM25 keyword index in RAM.
3. The UI will refresh, and you are ready to start querying the chatbot.

---

## 🧹 Troubleshooting & Management

### Stopping the App
Press `Ctrl + C` in the terminal terminal where Streamlit is running to stop the web server.

### Resetting Cache & Database
* **Clear Semantic Cache**: Click the **Clear Semantic Cache** button in the sidebar to purge past question memory.
* **Wipe Database completely**: Delete the `chroma_db/` folder and the `documents/` folder in the project root:
  * **Linux/macOS**:
    ```bash
    rm -rf chroma_db documents
    ```
  * **Windows (CMD)**:
    ```cmd
    rmdir /s /q chroma_db documents
    ```
  * **Windows (PowerShell)**:
    ```powershell
    Remove-Item -Recurse -Force chroma_db, documents
    ```
    *(The app will re-create them on the next run).*
