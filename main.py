import concurrent.futures as cf
import glob
import io
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Literal

import gradio as gr
import sentry_sdk
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger
from openai import OpenAI
from promptic import llm
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader
from tenacity import retry, retry_if_exception_type


if sentry_dsn := os.getenv("SENTRY_DSN"):
    sentry_sdk.init(sentry_dsn)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


class DialogueItem(BaseModel):
    text: str
    speaker: Literal["host-female", "main-speaker", "guest-1", "guest-2", "guest-3", "guest-4"]

    @property
    def voice(self):
        return {
            "host-female": "nova",      # Clear, professional female host voice
            "main-speaker": "onyx",        # Deep, authoritative male host voice
            "guest-1": "alloy",         # Balanced, neutral voice
            "guest-2": "echo",          # Younger-sounding voice
            "guest-3": "fable",         # Warm, welcoming voice
            "guest-4": "sage",          # Mature, knowledgeable-sounding voice
        }[self.speaker]


class Dialogue(BaseModel):
    scratchpad: str
    dialogue: List[DialogueItem]


def get_mp3(text: str, voice: str, api_key: str = None) -> bytes:
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )

    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text,
    ) as response:
        with io.BytesIO() as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
            return file.getvalue()


def generate_audio(file: str, openai_api_key: str = None) -> bytes:

    if not (os.getenv("OPENAI_API_KEY") or openai_api_key):
        raise gr.Error("OpenAI API key is required")

    with Path(file).open("rb") as f:
        reader = PdfReader(f)
        text = "\n\n".join([page.extract_text() for page in reader.pages])

    @retry(retry=retry_if_exception_type(ValidationError))
    @llm(
        model="gemini/gemini-1.5-flash-002",
        api_key="AIzaSyD2udCMKiixztGCp0gcpB2FDSlKE9s2ypE"
    )
    def generate_dialogue(text: str) -> Dialogue:
        """
        Your task is to take the input text provided and turn it into an engaging, dramatic podcast dialogue featuring four distinct voices discussing interconnected books and ideas. The input text may be messy or unstructured, as it could come from various sources like PDFs or web pages. Focus on extracting key points and creating compelling conversations that reveal hidden patterns and unexpected connections.

        Here is the input text you will be working with:

        <input_text>
        {text}
        </input_text>

        Important Formatting Rules:
        1. Do not use any bracketed placeholders or sound effects (e.g., no [Host], [Guest], or (sound effects))
        2. Match voice gender to speaker gender - use male voices for male authors and female voices for female authors
        3. Write dialogue to be read aloud directly - it will be converted to audio without modification
        4. Begin each line with the speaker's actual name (e.g., "Reada:" or "John Thompson:")
        5. Keep dialogue natural and conversational - avoid any formatting or stage directions

        Speaker Roles:
        - Reada (Host): Guides the conversation, draws connections, and asks probing questions
        - Main Speaker: Author of the core pattern book, leading the primary discussion
        - Guest 1 & 2: Authors of supporting books, offering complementary perspectives
        
        First, carefully read through the input text and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for an audio podcast.

        <scratchpad>
        Brainstorm creative ways to structure the dialogue that:
        1. Reveal hidden patterns in everyday experiences
        2. Connect common situations to unexpected domains
        3. Balance intriguing premises with practical relevance
        4. Vary narrative approaches (evolution, analysis, revelation)

        Consider using:
        - Analogies and metaphors that bridge familiar and novel concepts
        - Storytelling techniques that build dramatic tension
        - Hypothetical scenarios that make abstract ideas concrete
        - Strategic questioning that leads to moments of revelation
        - Contrasting viewpoints that create engaging dynamics

        Keep in mind that your podcast should be:
        - Accessible to a general audience while maintaining depth
        - Dramatically engaging while remaining informative
        - Structured to build toward key insights
        - Balanced between theoretical understanding and practical application

        Write your brainstorming ideas and outline here, noting:
        - Key dramatic moments and revelations
        - Points of connection between different authors' perspectives
        - Building blocks for narrative tension
        - Core insights and practical takeaways
        </scratchpad>

        Now create the podcast dialogue, incorporating:
        - Natural conversational flow between all speakers
        - Clear character voices and perspectives
        - Building dramatic tension and revelation
        - Engaging explanations of complex topics
        - Strategic integration of different viewpoints

        <podcast_dialogue>
        Write your engaging, dramatic podcast dialogue here. The conversation should:
        - Begin with a compelling hook that connects to everyday experience
        - Gradually reveal deeper patterns and unexpected connections
        - Build dramatic tension through contrasting viewpoints and revelations
        - Maintain practical relevance while exploring abstract concepts
        - End with a natural synthesis of key insights

        Make the dialogue as long and detailed as possible, while maintaining:
        - Clear speaker identities (use real author names)
        - Engaging conversational flow
        - Building dramatic tension
        - Natural integration of key concepts
        - Practical applications and takeaways

        At the end of the dialogue, have the host and guest speakers naturally summarize the main insights and takeaways from their discussion. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas one last time before signing off.
        </podcast_dialogue>
        """

    llm_output = generate_dialogue(text)

    audio = b""
    transcript = ""

    characters = 0

    with cf.ThreadPoolExecutor() as executor:
        futures = []
        for line in llm_output.dialogue:
            transcript_line = f"{line.speaker}: {line.text}"
            future = executor.submit(get_mp3, line.text, line.voice, openai_api_key)
            futures.append((future, transcript_line))
            characters += len(line.text)

        for future, transcript_line in futures:
            audio_chunk = future.result()
            audio += audio_chunk
            transcript += transcript_line + "\n\n"

    logger.info(f"Generated {characters} characters of audio")

    temporary_directory = "./gradio_cached_examples/tmp/"
    os.makedirs(temporary_directory, exist_ok=True)

    # we use a temporary file because Gradio's audio component doesn't work with raw bytes in Safari
    temporary_file = NamedTemporaryFile(
        dir=temporary_directory,
        delete=False,
        suffix=".mp3",
    )
    temporary_file.write(audio)
    temporary_file.close()

    # Delete any files in the temp directory that end with .mp3 and are over a day old
    for file in glob.glob(f"{temporary_directory}*.mp3"):
        if os.path.isfile(file) and time.time() - os.path.getmtime(file) > 24 * 60 * 60:
            os.remove(file)

    return temporary_file.name, transcript


demo = gr.Interface(
    title="PDF to Podcast",
    description=Path("description.md").read_text(),
    fn=generate_audio,
    examples=[[str(p)] for p in Path("examples").glob("*.pdf")],
    inputs=[
        gr.File(
            label="PDF",
        ),
        gr.Textbox(
            label="OpenAI API Key",
            visible=not os.getenv("OPENAI_API_KEY"),
        ),
    ],
    outputs=[
        gr.Audio(label="Audio", format="mp3"),
        gr.Textbox(label="Transcript"),
    ],
    allow_flagging="never",
    clear_btn=None,
    head=os.getenv("HEAD", "") + Path("head.html").read_text(),
    cache_examples="lazy",
    api_name=False,
)


demo = demo.queue(
    max_size=20,
    default_concurrency_limit=20,
)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    demo.launch(show_api=False)
