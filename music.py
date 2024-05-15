import logging
from typing import List, Dict, Union
from youtube_search import YoutubeSearch
from pydantic import BaseModel
from yt_dlp import YoutubeDL
from helpers import ROOT
from rag import get_music_chain
from models import get_ollama_local_model, LLM_FN, LLM
from json import load as json_load, dump as json_dump

# Constants
USE_CLIPBOARD = False
DEFAULT_DATA = {
    "title": "The Less I Know The Better",
    "artist": "Tame Impala",
    "year": 2015
}
DEFAULT_INPUT = "# Used to You • . • Ali Gatie 2019"
DEFAULT_LLM_FN = LLM_FN(get_ollama_local_model)
MANIFEST_FILE = "music_manifest.json"
SAVE_DIR = "YouTubeAudio"
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

def create_query(data: SearchSchema) -> str:
    return f"{data.title} {data.artist} official"

def query_to_top_youtube_hit(query: str) -> dict:
    results = YoutubeSearch(query, max_results=1).to_dict()
    if not results:
        raise ValueError("No results found")
    return results[0]

def url_from_top_hit(top_hit: dict) -> str:
    video_id = top_hit.get("id")
    if not video_id or len(video_id) != 11:
        raise ValueError("Invalid video ID")
    return f"{YOUTUBE_URL_PREFIX}{video_id}"

def download_url(url: str, save_dir: str):
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
    query = create_query(data)
    top_hit = query_to_top_youtube_hit(query)
    url = url_from_top_hit(top_hit)
    download_url(url, save_dir)

def append_to_music_manifest(data: List[Dict[str, Union[str, int]]], manifest_file: str = MANIFEST_FILE):
    filepath = ROOT / manifest_file
    if not filepath.exists():
        with open(filepath, "w") as f:
            json_dump([], f)
    with open(filepath, "r") as f:
        manifest = json_load(f) or []
    manifest.extend(data)
    with open(filepath, "w") as f:
        json_dump(manifest, f)

def music_workflow(query: str) -> bool:
    llm = LLM(DEFAULT_LLM_FN).llm
    chain = get_music_chain(llm)
    res = chain.invoke(query)
    if not res:
        logger.info("No results found")
        return False
    if not isinstance(res, list):
        raise TypeError("res must be a list")
    append_to_music_manifest(res)
    return True

def download_from_manifest(manifest_file: str = MANIFEST_FILE, save_dir: str = SAVE_DIR):
    filepath = ROOT / manifest_file
    if not filepath.exists():
        logger.info("Manifest file does not exist")
        return
    with open(filepath, "r") as f:
        manifest = json_load(f) or []
    if not manifest:
        logger.info("Music manifest is empty")
        return
    for item in manifest:
        download_audio(SearchSchema(**item), save_dir)
    with open(filepath, "w") as f:
        json_dump([], f)

def main():
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
        download_from_manifest(manifest_file, save_dir)

if __name__ == "__main__":
    main()