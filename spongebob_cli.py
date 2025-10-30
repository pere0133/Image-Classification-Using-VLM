#!/usr/bin/env python3
"""
Example: One-shot call to ai.sooners.us using gemma3:4b model.

Requires:
  pip install requests python-dotenv
"""

import os
import requests
from dotenv import load_dotenv

# Load API key and base URL from ~/.soonerai.env
load_dotenv(os.path.join(os.path.expanduser("~"), ".soonerai.env"))

API_KEY = os.getenv("SOONERAI_API_KEY")
BASE_URL = os.getenv("SOONERAI_BASE_URL", "https://ai.sooners.us").rstrip("/")
MODEL = os.getenv("SOONERAI_MODEL", "gemma3:4b")

if not API_KEY:
    raise RuntimeError("Missing SOONERAI_API_KEY in ~/.soonerai.env")

# Prepare API request
url = f"{BASE_URL}/api/chat/completions"

history = [{"role": "system", "content": "You are SpongeBob SquarePants. Speak cheerfully and use ocean humor."}]

print("SpongeBob is ready to talk to you! Type 'exit' to end this program.\n")  #for the user

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Adiós amigo, goodbye!")
        break

    history.append({"role": "user", "content": user_input})

    payload = {
        "model": MODEL,
        "messages": history,
        "temperature": 0.6,
    }

    # Send POST request
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    # Display result
    if response.status_code == 200:
        data = response.json()
        reply = data["choices"][0]["message"]["content"]
        print("SpongeBob:", reply)

        # SpongeBob response to history
        history.append({"role": "assistant", "content": reply})

        # OPTIONAL PART: Keeping only last 10 messages to avoid long context
        if len(history) > 12:  # 1 system + last 5 user turns + 5 bot turns
            history = [history[0]] + history[-10:]

    else:
        print(f"Error {response.status_code}: {response.text}")

