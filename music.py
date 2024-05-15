from youtube_search import YoutubeSearch
from pydantic import BaseModel
from yt_dlp import YoutubeDL
from helpers import ROOT
from rag import get_music_chain
from models import get_together_llama3, get_ollama_local_model, LLM_FN, LLM
from json import load as json_load, dump as json_dump

USE_CLIPBOARD = False
DEFAULT_DATA = {
    "title": "The Less I Know The Better",
    "artist": "Tame Impala",
    "year": 2015
}
# DEFAULT_INPUT = "The Less I Know The Better Tame Impala 2015"
DEFAULT_INPUT = "# Used to You • . • Ali Gatie 2019"
DEFAULT_LLM_FN = LLM_FN(get_ollama_local_model)

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
    results = YoutubeSearch(query, 1)._search()
    # print(results)
    assert results, "No results found"
    # Extract the top URL
    top_hit = results[0]
    return top_hit

def url_from_top_hit(top_hit: dict) -> str:
    id = top_hit["id"]
    assert len(id) == 11, "Invalid video ID"
    url = "https://youtube.com/watch?v=" + top_hit["id"]
    return url

def download_url(url, save_dir):
    ydl_opts = {
        "format": "m4a/bestaudio/best",
        "noplaylist": True,
        "outtmpl": save_dir + "/%(title)s.%(ext)s",
        "youtube_include_dash_manifest": False,
        "youtube_include_hls_manifest": False,
    }
    assert isinstance(url, str), "url must be a string"
    assert url.startswith("https://youtube.com"), "Invalid URL"
    urls = [url]
    # Download file
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)

def download_audio(data: SearchSchema, save_dir: str):
    query = create_query(data)
    top_hit = query_to_top_youtube_hit(query)
    url = url_from_top_hit(top_hit)
    download_url(url, save_dir)

def append_to_music_manifest(data: list[dict], manifest_file: str = "music_manifest.json"):
    assert isinstance(data, list), "data must be a list"
    # assert manifest file exists, if not create it
    filepath = ROOT / manifest_file
    if not filepath.exists():
        with open(filepath, "w") as f:
            json_dump([], f)
            return
    with open(filepath, "r") as f:
        manifest = json_load(f)
    if not manifest:
        print("music manifest is empty")
    for item in data:
        manifest.append(item)
    with open(filepath, "w") as f:
        json_dump(manifest, f)

def music_workflow(query: str) -> bool:
    llm = LLM(DEFAULT_LLM_FN).llm
    chain = get_music_chain(llm)
    res = chain.invoke(query)
    if not res:
        print("No results found")
        return False
    assert isinstance(res, list), "res must be a list"
    append_to_music_manifest(res)
    return True

def download_from_manifest(manifest_file: str = "music_manifest.json", save_dir: str = "YouTubeAudio"):
    filepath = ROOT / manifest_file
    if not filepath.exists():
        print("manifest.json does not exist")
        return
    with open(filepath, "r") as f:
        manifest = json_load(f)
    if not manifest:
        print("music manifest is empty")
        return
    for item in manifest:
        download_audio(SearchSchema(**item), save_dir)
    # Clear the manifest
    with open(filepath, "w") as f:
        json_dump([], f)

def main():
    # Given data that matches the schema, create a query
    # data = {
    #     "title": "Used to You",
    #     "artist": "Ali Gatie",
    #     "album": "You",
    #     "year": 2019
    # }

    # data = SearchSchema(**data)
    # query = create_query(data)
    # top_hit = query_to_top_youtube_hit(query)
    # url = url_from_top_hit(top_hit)
    # print(url)
    # save_dir = "YouTubeAudio"
    # download_url(url, save_dir)
    # download_from_manifest()
    manifest_file = "music_manifest.json"
    save_dir = "YouTubeAudio"
    if USE_CLIPBOARD:
        from pyperclip import paste
        INPUT = paste()
    else:
        INPUT = DEFAULT_INPUT
    found_songs = music_workflow(INPUT)
    if found_songs:
        print("Downloading songs...")
        download_from_manifest(manifest_file, save_dir)

if __name__ == "__main__":
    main()
