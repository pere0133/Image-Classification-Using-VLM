## 🧽 SpongeBob CLI Chatbot  
**Perspectives on Computing — GenAI Tools Assignment**  
This project implements a **SpongeBob-style command-line chatbot** using the **OU SoonerAI API** (`https://ai.sooners.us`) and the `gemma3:4b` model.  
It maintains **multi-turn chat history**, loads API keys from a secure `.env` file, and runs locally in a Python terminal environment.

---

### 🚀 Features
- 🌊 Talks like SpongeBob SquarePants  
- 💬 Multi-turn chat history (system + previous messages)  
- 🔐 Secure API key handling via `~/.soonerai.env`  
- 🤖 Uses OpenAI-compatible `/api/chat/completions` endpoint  
- 🎯 Powered by **gemma3:4b** on **ai.sooners.us**

---

### 🛠️ Requirements
- Python 3.8+
- `requests`, `python-dotenv`

Install dependencies:

```bash
pip install requests python-dotenv
```

Or using your virtual environment:

```bash
pip freeze > requirements.txt
```

---

### 🔐 Environment Setup

Create a file **in your home directory** (not the project folder):

**Windows PowerShell:**

```bash
notepad $env:USERPROFILE\.soonerai.env
```

Paste and save:

```
SOONERAI_API_KEY=your_api_key_here
SOONERAI_BASE_URL=https://ai.sooners.us
SOONERAI_MODEL=gemma3:4b
```

> ✅ This file must **not** be pushed to GitHub.

---

### ▶️ Running the Chatbot

Inside your repo folder / venv:

```bash
python spongebob_cli.py
```

Type messages and chat!  
Type `exit` to quit.

---

### 💬 Example Conversation

```
SpongeBob is ready to talk to you! Type 'exit' to end this program.

You: hi there
SpongeBob: Hiiiiii there, buddy! It's me, SpongeBob SquarePants! Isn’t it just a *splashtastic* day?! The water's so bubbly and the sunshine's tickling my porous side! Do you like jellyfishing? It’s my absolute FAVORITE! ...

You: do you speak spanish?
SpongeBob: ¡Hola, amigo! ... ¡Vamos! (Let’s go!)

You: exit
Adiós amigo, goodbye!
```

---

### 📁 Repo File Structure

```
spongebob_cli.py
README.md
requirements.txt
.gitignore
```

`.gitignore` should include:

```
__pycache__/
venv/
*.env
*.soonerai.env
```

---

### 🧠 Credits
- OU SoonerAI platform  
- Assignment by **Perspectives on Computing**
- Readme built using OpenAI ChatGPT