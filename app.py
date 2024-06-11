from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
import time
import spotify
import os
from models import get_ollama_mistral, LLM_FN, LLM, get_together_llama3
# from mutagen.mp4 import MP4
from langchain.schema import SystemMessage, AIMessage, HumanMessage
#
app = FastAPI()

# MUSIC_FILEPATH = "musicdir"
# This file is in the folder GitHub/Augmenta. Make the music filepath
# GitHub/AugmentaMusic/HQMusic
MUSIC_FILEPATH = "/Users/tanujvasudeva/Documents/GitHub/AugmentaMusic/HQMusic"


def stream_llm_response(query, llm, is_Ollama=False):
    messages = [
        SystemMessage(content="You are a helpful AI."),
        HumanMessage(content=query)
    ]
    if is_Ollama:
        for chunk in llm.stream(messages):
            yield chunk
    else:
        for chunk in llm.stream(messages):
            yield chunk.content


@app.get("/v1/chat/query={query}")
async def chat(query="What is a savory recipe with fruits?"):
    llm = LLM(LLM_FN(get_together_llama3))
    is_ollama = False
    # return {}
    return StreamingResponse(
        stream_llm_response(
            query,
            llm,
            is_ollama),
        media_type="text/plain")


@app.get("/v1/stream")
async def stream_audio_file(filename: str) -> StreamingResponse:
    SHORTCUTS = {"cough": "Young the Giant - Cough Syrup"}
    if filename in SHORTCUTS:
        filename = SHORTCUTS[filename]
    if filename[-4:] != ".m4a":
        print("Adding .m4a extension")
        filename = filename + ".m4a"
    filepath = f"{MUSIC_FILEPATH}/{filename}"

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    def iterfile():
        with open(filepath, mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile(), media_type="audio/mp4")


@app.get("/v1/movie")
async def stream_movie(filename: str = "Modern.Family.S01E01.mp4"):
    filepath = f"{MUSIC_FILEPATH}/{filename}"

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    def iterfile():
        with open(filepath, mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile(), media_type="video/mp4")


@app.get("/download/{filename}")
async def download_file(filename: str):
    filepath = f"{MUSIC_FILEPATH}/{filename}"

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(filepath, media_type="audio/mp4", filename=filename)

# def get_metadata(file_path):
#     audio = MP4(file_path)
#     metadata = {
#         "title": audio.get("\xa9nam", ["Unknown"])[0],
#         "artist": audio.get("\xa9ART", ["Unknown"])[0],
#         "album": audio.get("\xa9alb", ["Unknown"])[0],
#         "genre": audio.get("\xa9gen", ["Unknown"])[0],
#         "year": audio.get("\xa9day", ["Unknown"])[0],
#     }
#     return metadata


def test_spotify_function() -> dict[str, str]:
    # return {"message": "Server is alive!"}
    return spotify.guess_album_name_from_song_name("My heart will go on")

# Create a route that calls the Python function
@app.get("/v1/test")
async def test_endpoint():
    response = test_spotify_function()
    return response


@app.get("/v1/actions/play")
async def play():
    # Do something to play music
    start_playback = None
    if start_playback:
        return {"message": "Started playback on device"}
    else:
        return {"message": "Failed to play music"}

# Define a generator function that yields data


def my_streaming_function():
    repeat = 8
    for i in range(repeat):
        with open("sample.txt", "r") as file:
            for line in file:
                yield line
                time.sleep(0.05)  # Simulate a delay

# Create a route that streams the response


@app.get("/v1/stream-text")
async def stream_endpoint():
    # create_playlist("api")
    return StreamingResponse(my_streaming_function(), media_type="text/plain")
