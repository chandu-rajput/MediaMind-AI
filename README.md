# MediaMind AI 🎙️

An AI-powered video and audio assistant that transforms YouTube videos and uploaded recordings into structured notes and interactive conversations.

> Add a screenshot or short screen recording of the app here.

## Features

- **Audio & Video Input:** Process YouTube videos and uploaded audio/video files.
- **Speech Transcription:** Convert speech into text locally using OpenAI Whisper.
- **Multilingual Support:** Transcribe English audio, and translate Hindi and Hinglish audio into English text.
- **Meeting Summarization:** Generate concise summaries using Mistral AI.
- **Information Extraction:** Extract action items, key decisions, and open questions.
- **RAG-Based Chat:** Ask questions about video or meeting transcripts using Retrieval-Augmented Generation.
- **Export Notes:** Download meeting notes as TXT files.

## How It Works

```
YouTube URL / uploaded file
        |
  yt-dlp + Pydub  ->  16 kHz mono WAV, split into 5-minute chunks
        |
  Whisper (local)  ->  transcript (translated to English for Hindi/Hinglish)
        |
        +--> Mistral via LangChain: summary, title, action items,
        |    key decisions, open questions
        |
        +--> transcript chunks -> Hugging Face embeddings -> ChromaDB
                    |
          question -> retriever (top 4 chunks) -> Mistral -> answer
```

- **Summarization** uses map-reduce: each transcript chunk is summarized separately, then the partial summaries are merged into one structured summary.
- **RAG chat:** the question is embedded with the same model used for the transcript, the 4 most similar chunks are retrieved from ChromaDB, and Mistral answers using only that context.

## Tech Stack

- **Language:** Python
- **LLM:** Mistral AI
- **Speech Recognition:** OpenAI Whisper (runs locally)
- **Framework:** LangChain
- **Embeddings:** Hugging Face (`all-MiniLM-L6-v2`)
- **Vector Database:** ChromaDB
- **Interface:** Streamlit
- **Audio Processing:** FFmpeg, Pydub, yt-dlp

## Project Structure

```
MediaMind-AI/
│
├── core/
│   ├── llm.py
│   ├── transcript.py
│   ├── summarize.py
│   ├── extractor.py
│   ├── vector_store.py
│   └── rag_engine.py
│
├── utils/
│   └── audio_processor.py
│
├── downloads/          (created automatically)
├── main.py
├── app.py
├── .env.example
├── .gitignore
├── pyproject.toml
└── uv.lock
```

## Installation

### 1. Install uv and FFmpeg

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. FFmpeg is required for audio processing and must be available on your system PATH.

```bash
# Windows
choco install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

### 2. Clone the repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/mediamind-ai.git
cd mediamind-ai
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Configure environment variables

Copy the example file and add your key:

```bash
# macOS / Linux
cp .env.example .env

# Windows
copy .env.example .env
```

Then edit `.env`:

```
MISTRAL_API_KEY=your_mistral_api_key
```

## Usage

Run the Streamlit application:

```bash
uv run streamlit run app.py
```

Or run the terminal-based version:

```bash
uv run python main.py
```

The first run downloads the Whisper and embedding models, so it needs an internet connection and takes a little longer.

## API Requirements

MediaMind AI uses the Mistral AI API for summarization, information extraction, and RAG-based question answering. API usage is subject to the selected model's rate limits and account quotas. You can switch to another Mistral model in `core/llm.py`.

## Known Limitations

- **Free-tier rate limits:** the free Mistral API limits requests per minute, so long recordings can trigger `429` errors. Short recordings (around 10 minutes) work best.
- **Designed for short recordings:** audio is processed in 5-minute chunks and summarized in a single map-reduce pass.
- **One recording at a time:** processing a new recording replaces the previous vector store.
- **No chat memory:** each question is answered independently, so follow-ups like "when is it due?" do not know what "it" refers to.
- **Whisper on CPU is slow:** a GPU or a smaller Whisper model speeds this up.
- **No speaker labels:** the transcript does not say who said what.

## Future Improvements

- PDF export for meeting notes.
- Hierarchical summarization for long recordings.
- Conversation history in the chat.
- Speaker diarization.
- More flexible LLM provider configuration.

