import logging
import os
from pathlib import Path
from typing import List, Dict, Union
from json import JSONDecodeError, load as json_load, dump as json_dump

from youtube_search import YoutubeSearch
from pydantic import BaseModel
from yt_dlp import YoutubeDL

from helpers import ROOT
from rag import get_eval_chain, get_music_chain
from models import get_ollama_local_model, LLM_FN, LLM, get_together_fn_mix

# Configuration flags
DISABLE_DOWNLOADING = True

USE_LOCAL_MODEL = False
CONFIRM_SCHEMA_TYPE = True
USE_CLIPBOARD = True
ONLY_DOWNLOAD = False

QUERY_SUFFIX = "official audio"

DEFAULT_INPUT = "fleetwood mac landslide"
DEFAULT_LLM_FN = LLM_FN(
    get_ollama_local_model) if USE_LOCAL_MODEL else LLM_FN(get_together_fn_mix)
MANIFEST_FILE = "music_db.json"
MUSIC_DIR = ROOT.parent / "AugmentaMusic"
SAVE_DIR = MUSIC_DIR / "YouTubeAudio"
YOUTUBE_URL_PREFIX = "https://youtube.com/watch?v="

EXAMPLE_JSON_OUTPUT_STRING = """[{"title": "Right Where I Belong", "artist": "Alex Goot", "album": "Wake Up Call"},
{"title": "Mockingbird", "artist": "Eminem", "album": "Encore (Deluxe Version)"},
{"title": "Wicked Man's Rest", "artist": "Passenger", "album": "Wicked Man's Rest"},
{"title": "Martin & Gina", "artist": "Polo G", "album": "THE GOAT"},
{"title": "Le Festin", "artist": "Camille", "album": "Ratatouille (Score from the Motion Picture)"},
{"title": "In da Club", "artist": "50 Cent", "album": "Get Rich or Die Tryin'"},
{"title": "Thanos vs J. Robert Oppenheimer", "artist": "Epic Rap Battles of History", "album": "Thanos vs J. Robert Oppenheimer - Single"}]"""

REQUIRED_KEYS = ["title", "artist", "album"]


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Few-shot examples for the language model
DEFAULT_FEW_SHOT_EXAMPLES = [
    {
        "input": "drivers license E Olivia Rodrigo SOUR (Video Version)\nSTAY E The Kid LAROI, Justin Bieber F*CK LOVE 3+: OVER YOU",
        "output": '[{"title": "drivers license", "artist": "Olivia Rodrigo", "album": "SOUR (Video Version)"}, {"title": "STAY", "artist": "The Kid LAROI, Justin Bieber", "album": "F*CK LOVE 3+: OVER YOU"}]'
    },
    {
        "input": "Right Where I Belong • .. Alex Goot Wake Up Call Mockingbird 0 ... Eminem Encore (Deluxe Version) Wicked Man's Rest Passenger Wicked Man's Rest Martin & Gina E... Polo G THE GOAT Le Festin ... Camille Ratatouille (Score from the Motion Picture) In da Club ... 50 Cent Get Rich or Die Tryin' Thanos vs J. Robert Oppenheimer • . . Epic Rap Battles of History Thanos vs J. Robert Oppenheimer - Single",
        "output": EXAMPLE_JSON_OUTPUT_STRING
    }
]
# DEFAULT_FEW_SHOT_EXAMPLES = None

# Pydantic models for data validation


class SearchSchema(BaseModel):
    title: str
    artist: str
    album: str


class ResultSchema(BaseModel):
    url: str
    title: str


class FinalSchema(BaseModel):
    title: str
    artist: str
    # album: str
    # year: int
    downloaded: bool
    url: str


class AppleMusicSchema(BaseModel):
    title: str
    artist: str
    album: str


def create_query(data: SearchSchema) -> str:
    """Create a search query from the given data."""
    return f"{data.title} {data.artist} {QUERY_SUFFIX}"


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


def download_url(url: str, save_dir: str, force_m4a: bool = False):
    """Download audio from the given URL."""
    ydl_opts = {
        "format": "m4a/bestaudio/best",
        "noplaylist": True,
        "outtmpl": f"{save_dir}/%(title)s.%(ext)s",
        "youtube_include_dash_manifest": False,
        "youtube_include_hls_manifest": False,
        "extractor_args": {'youtube': {'player_client': ['web']}}
    }
    if force_m4a:
        ydl_opts['postprocessors'] = [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }]
    if not url.startswith(YOUTUBE_URL_PREFIX):
        raise ValueError("Invalid URL")
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def download_audio(data: SearchSchema, save_dir: str,
                   manifest_ids: List[str] = []) -> Union[None, str]:
    """Download audio for the given search data."""
    query = create_query(data)
    top_hit = query_to_top_youtube_hit(query)
    url = url_from_top_hit(top_hit)
    video_id = url.split("v=")[1]
    if video_id in manifest_ids:
        logger.info("Skipping download for repeat YouTube ID!")
        return url
    download_url(url, save_dir)
    return url


def append_to_music_manifest(
        data: List[Dict[str, Union[str, int]]], manifest_file: str = MANIFEST_FILE) -> bool:
    """Append data to the music manifest file."""
    if CONFIRM_SCHEMA_TYPE:
        for item in data:
            if not all(key in item for key in REQUIRED_KEYS):
                raise ValueError(
                    "At least one song does not contain the required keys")
            SearchSchema(**item)
            item["downloaded"] = False

    filepath = ROOT / manifest_file
    if not filepath.exists():
        filepath.write_text("[]")

    try:
        with filepath.open("r") as f:
            manifest = json_load(f) or []
    except JSONDecodeError:
        logger.error("Invalid JSON in manifest file")
        manifest = []

    if CONFIRM_SCHEMA_TYPE:
        for item in manifest:
            SearchSchema(**item)

    # TODO: Check for duplicates more effectively
    existing_items = {
        (item["title"], item["artist"], item["album"])
        for item in manifest}

    new_items = [
        item for item in data if (
            item["title"],
            item["artist"],
            item["album"]) not in existing_items]
    if not new_items:
        logging.info("No new items to add")
        return False

    manifest.extend(new_items)
    with filepath.open("w") as f:
        json_dump(manifest, f, indent=2)
    return True


def music_workflow(query: str) -> bool:
    """Run the music workflow for the given query."""
    llm = LLM(DEFAULT_LLM_FN).llm
    few_shot_examples = DEFAULT_FEW_SHOT_EXAMPLES

    eval_chain = get_eval_chain(llm)
    eval_dict = {
        "excerpt": "index 0:\n" + query,
        "criteria": "The excerpt contains at least one song title, artist, and album."}
    res = eval_chain.invoke(eval_dict)
    if not res["meetsCriteria"]:
        logger.info("Excerpt does not meet criteria")
        return False

    chain = get_music_chain(llm, few_shot_examples)
    response_object = chain.invoke(query)
    if not response_object:
        logger.info("No results found")
        return False
    if not isinstance(response_object, list):
        raise TypeError("Response object must be a list")

    return append_to_music_manifest(response_object)


def download_from_manifest(
        manifest_file: str = MANIFEST_FILE,
        save_dir: str = SAVE_DIR):
    """Download audio files from the music manifest."""
    filepath = ROOT / manifest_file
    if not filepath.exists():
        logger.info("Manifest file does not exist")
        return

    try:
        with filepath.open("r") as f:
            manifest = json_load(f)
    except JSONDecodeError:
        logger.error("Invalid JSON in manifest file")
        return

    if not manifest:
        logger.info("Music manifest is empty")
        return

    manifest_ids = [item["url"].split("v=")[1] for item in manifest if item.get(
        "url") and item.get("downloaded")]

    for item in manifest:
        if item.get("downloaded"):
            continue
        url = download_audio(SearchSchema(**item), save_dir, manifest_ids)
        if not url:
            logger.error("Failed to download audio")
            continue
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
    only_download = ONLY_DOWNLOAD

    if only_download:
        if DISABLE_DOWNLOADING:
            logging.error(
                "Downloading is disabled but ONLY_DOWNLOAD is enabled. Aborting.")
            return
        download_from_manifest(manifest_file, save_dir)
        return

    input = DEFAULT_INPUT

    if USE_CLIPBOARD:
        try:
            from pyperclip import paste
            clipboard_contents = paste().strip()
            if not clipboard_contents:
                logger.info("Clipboard is empty. Using default input.")
            else:
                logger.info("Using clipboard contents as input")
                input = clipboard_contents
        except ImportError:
            logger.error(
                "pyperclip is not installed. Falling back to default input.")

    assert len(input) > 0
    try:
        found_songs = music_workflow(input)
    except KeyboardInterrupt:
        logger.info("Music workflow aborted")
        return

    if found_songs:
        logger.info("Songs found and added to manifest")
        if not DISABLE_DOWNLOADING:
            logger.info("Downloading songs...")
            download_from_manifest(manifest_file, save_dir)


if __name__ == "__main__":
    main()
