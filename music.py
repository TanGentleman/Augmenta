import logging
import os
from pathlib import Path
from typing import List, Dict, Union
from json import JSONDecodeError, load as json_load, dump as json_dump

from youtube_search import YoutubeSearch
from pydantic import BaseModel
from yt_dlp import YoutubeDL

from helpers import ROOT
from rag import get_music_chain
from models import get_ollama_local_model, LLM_FN, LLM, get_together_llama3

USE_LOCAL_MODEL = True
CONFIRM_SCHEMA_TYPE = True
# Constants
USE_CLIPBOARD = True
DEFAULT_DATA = {
    "title": "The Less I Know The Better",
    "artist": "Tame Impala",
    "year": 2015
}
DEFAULT_INPUT = "# Used to You • . • Ali Gatie 2019"
if USE_LOCAL_MODEL:
    DEFAULT_LLM_FN = LLM_FN(get_ollama_local_model)
else:
    DEFAULT_LLM_FN = LLM_FN(get_together_llama3)
MANIFEST_FILE = "music_db.json"
MUSIC_DIR = ROOT.parent / "AugmentaMusic"
SAVE_DIR = MUSIC_DIR / "YouTubeAudio"
YOUTUBE_URL_PREFIX = "https://youtube.com/watch?v="

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchSchema(BaseModel):
    title: str
    artist: str
    year: int

class ResultSchema(BaseModel):
    url: str
    title: str

class FinalSchema(BaseModel):
    title: str
    artist: str
    year: int
    downloaded: bool
    url: str

def create_query(data: SearchSchema) -> str:
    """Create a search query from the given data."""
    return f"{data.title} {data.artist} official"

def query_to_top_youtube_hit(query: str) -> dict:
    """Get the top YouTube hit for the given query."""
    results = YoutubeSearch(query, max_results=1).to_dict()
    if not results:
        raise ValueError("No results found")
    return results[0]

def url_from_top_hit(top_hit: dict) -> str:
    """Extract the URL from the top YouTube hit."""
    video_id = top_hit.get("id")
    if not video_id or len(video_id) != 11:
        raise ValueError("Invalid video ID")
    return f"{YOUTUBE_URL_PREFIX}{video_id}"

def download_url(url: str, save_dir: str):
    """Download audio from the given URL."""
    ydl_opts = {
        "format": "m4a/bestaudio/best",
        "noplaylist": True,
        "outtmpl": f"{save_dir}/%(title)s.%(ext)s",
        "youtube_include_dash_manifest": False,
        "youtube_include_hls_manifest": False,
    }
    if not url.startswith(YOUTUBE_URL_PREFIX):
        raise ValueError("Invalid URL")
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def download_audio(data: SearchSchema, save_dir: str):
    """Download audio for the given search data."""
    query = create_query(data)
    top_hit = query_to_top_youtube_hit(query)
    url = url_from_top_hit(top_hit)
    download_url(url, save_dir)
    return url

def append_to_music_manifest(data: List[Dict[str, Union[str, int]]], manifest_file: str = MANIFEST_FILE) -> bool:
    """Append data to the music manifest file."""
    if CONFIRM_SCHEMA_TYPE:
        for item in data:
            SearchSchema(**item)
            item["downloaded"] = False

    filepath = ROOT / manifest_file
    if not filepath.exists():
        filepath.write_text("[]")

    try:
        with filepath.open("r") as f:
            manifest = json_load(f) or []
    except JSONDecodeError:
        print("Invalid JSON in manifest file")
        manifest = []

    if CONFIRM_SCHEMA_TYPE:
        for item in manifest:
            SearchSchema(**item)

    # Use a set to track existing items for faster duplicate checking
    existing_items = {(item["title"], item["artist"], item["year"]) for item in manifest}

    new_items = []
    for item in data:
        if (item["title"], item["artist"], item["year"]) not in existing_items:
            new_items.append(item)
            existing_items.add((item["title"], item["artist"], item["year"]))
        else:
            print("Duplicate entry found for song:", item["title"])
    if not new_items:
        return False
    manifest.extend(new_items)

    with filepath.open("w") as f:
        json_dump(manifest, f, indent=2)
    return True

def music_workflow(query: str) -> bool:
    """Run the music workflow for the given query."""
    llm = LLM(DEFAULT_LLM_FN).llm
    chain = get_music_chain(llm)
    response_object = chain.invoke(query)
    if not response_object:
        logger.info("No results found")
        return False
    if not isinstance(response_object, list):
        raise TypeError("res must be a list")
    res = append_to_music_manifest(response_object)
    if not res:
        logger.info("No new songs found")
        return False
    return True

def download_from_manifest(manifest_file: str = MANIFEST_FILE, save_dir: str = SAVE_DIR):
    """Download audio files from the music manifest."""
    filepath = ROOT / manifest_file
    if not filepath.exists():
        logger.info("Manifest file does not exist")
        return
    with filepath.open("r") as f:
        try:
            manifest = json_load(f)
        except JSONDecodeError:
            print("Invalid JSON in manifest file")
            manifest = []
    if not manifest:
        logger.info("Music manifest is empty")
        return
    for item in manifest:
        if item.get("downloaded"):
            continue
        url = download_audio(SearchSchema(**item), save_dir)
        item["downloaded"] = True
        item["url"] = url
        if CONFIRM_SCHEMA_TYPE:
            FinalSchema(**item)
    with filepath.open("w") as f:
        json_dump(manifest, f, indent=2)


def main():
    """Main function to run the script."""
    manifest_file = MANIFEST_FILE
    save_dir = SAVE_DIR

    if USE_CLIPBOARD:
        try:
            from pyperclip import paste
            INPUT = paste()
        except ImportError:
            logger.error("pyperclip is not installed. Falling back to default input.")
            INPUT = DEFAULT_INPUT
    else:
        INPUT = DEFAULT_INPUT

    found_songs = music_workflow(INPUT)
    if found_songs:
        logger.info("Downloading songs...")
        # download_from_manifest(manifest_file, save_dir)

if __name__ == "__main__":
    main()