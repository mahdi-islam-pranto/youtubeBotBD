# 📺 YT Bangla Chatbot

That lets you chat in Bangla with any YouTube video!  
Just paste a YouTube video link, and ask questions in Bangla — the bot will answer using the video’s transcript, even if the video is in another language.

---

## Features

-  **Paste any YouTube video link**
-  **Ask questions in Bangla**
-  **Works with videos in any language (auto-translation supported)**
-  **Retrieval-Augmented Generation (RAG) for accurate answers**
-  **Vector store for efficient retrieval**
-  **Modern, user-friendly chat interface**
-  **If the answer isn’t found, the bot asks a relevant follow-up question in Bangla**
-  **Beautiful dark theme and responsive layout**

---


### Home Page

![Home Page](screenshots/yt_bangla_chatbot1.png)

### Main Chat Interface

![Main Chat Interface](screenshots/yt_bangla_chatbot2.png)



---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/YoutubeBanglaBot.git
cd YoutubeBanglaBot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or, manually:

```bash
pip install streamlit langchain-openai langchain-community youtube-transcript-api python-dotenv
```

### 3. Set up your `.env` file

Create a `.env` file in the project root and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the app

```bash
streamlit run web_app2.py
```

---

##  How it Works

1. **Paste a YouTube video link**  
2. The app fetches and (if needed) translates the transcript  
3. The transcript is split and embedded for retrieval  
4. **Ask your question in Bangla**  
5. The bot finds the most relevant transcript chunks and generates a detailed answer in Bangla

---

##  Contact

[Mahdi Islam Pranto](https://www.linkedin.com/in/mahdi-islam-pranto/)


---

> **Note:**  
> This project uses OpenAI's GPT and embedding models. You need an OpenAI API key to use the chatbot.
