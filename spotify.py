import logging
from os import environ
from typing import Union, List, Tuple, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()

SPOTIFY_CLIENT_ID = environ.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = environ.get("SPOTIFY_CLIENT_SECRET")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SEARCH_MARKET = "US"
FILTER_DOMAIN = "site:open.spotify.com/"
ALLOW_AUTHORIZED = False
DEFAULT_QUERY = "2 poor kids"

# Spotify Clients
READ_ONLY_SP = Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET))

AUTHORIZED_SP = None
if ALLOW_AUTHORIZED:
    AUTHORIZED_SP = Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri="http://localhost:3000",
        scope="user-library-read user-library-modify"))


def perform_lookup(
        query: str, max_results: int = 3) -> list[dict[str, Union[str, int]]] | None:
    """Perform a lookup for the given query."""
    results = TavilySearchResults(max_results=max_results).invoke(query)
    if not results:
        logging.error("No Tavily results found")
        return None
    return results


def uris_from_spotify_urls(urls: list[str]) -> list[Tuple[str, str]] | None:
    """
    Given a list of Spotify URLs, extract the URI type and URI.

    Args:
        urls (list[str]): A list of Spotify URLs.

    Returns:
        list[Tuple[str, str]] | None: A list of tuples containing the URI type and URI.
    """
    uri_items = []
    for url in urls:
        if not url.startswith("https://open.spotify.com/"):
            logger.error("Invalid URL")
            continue
        uri_type = url.split("/")[3]
        uri = url.split("/")[-1].split("?")[0]
        if uri_type not in ["track", "playlist", "album"]:
            logger.info(f"Ignoring URL of type {uri_type}")
            continue
        uri_items.append((uri_type, uri))
    if not uri_items:
        logger.error("No valid objects found")
        return None
    return uri_items


def decode_uris(uri_items: str, expand_tracks=False, limit=10):
    """
    Given a list of uri types and uris, decode the Spotify URIs and print the tracks, playlists, and albums.

    Args:
        urls (str): A list of Spotify URLs.
        expand_playlist (bool, optional): If True, print tracks in the playlist. Defaults to False.
        limit (int, optional): The limit of tracks to show. Defaults to 10.

    Returns:
        dict: A dictionary containing the tracks, playlists, and albums.
    """
    results = {"tracks": [], "playlists": [], "albums": []}
    for uri_type, uri in uri_items:
        if uri_type not in ["track", "playlist", "album"]:
            logging.info("Ignoring uri of type", uri_type)
            continue
        if uri_type == "playlist":
            playlist_URI = uri
            playlist = READ_ONLY_SP.playlist(playlist_URI)
            playlist_name = playlist["name"]
            print("Playlist:", playlist_name)
            if expand_tracks:
                if len(playlist_tracks) > limit:
                    logging.info(f"{len(playlist_tracks)} tracks truncated to {limit}.")
                # Use playlist object to get tracks
                playlist_tracks = playlist["tracks"]["items"]
                print_items(playlist_tracks, from_playlist=True, limit=limit)
            results["playlists"].append(playlist)
        elif uri_type == "track":
            track = READ_ONLY_SP.track(uri)
            track_name = track["name"]
            artist_name = track["artists"][0]["name"]
            print(f"Track: {track_name} by {artist_name}")
            results["tracks"].append(track)
        elif uri_type == "album":
            album = READ_ONLY_SP.album(uri)
            results["albums"].append(album)
            album_name = album["name"]
            if expand_tracks:
                tracks = album["tracks"]["items"]
                if len(tracks) > limit:
                    logging.info(f"{len(tracks)} tracks truncated to {limit}.")
                print("Album:", album_name, "Tracks:")
                print("=======")
                for track in tracks:
                    print(track["name"])
                print()
            else:
                print("Album:", album_name)
        else:
            logging.info("Ignoring URL")
    if not any(results.values()):
        logging.error("No valid items found")
        return None
    return results


def search_spotify_tracks(query: str, limit: int = 2) -> dict | None:
    result = READ_ONLY_SP.search(
        q=query,
        limit=limit,
        type="track",
        market=SEARCH_MARKET)
    if result is None:
        logging.error("No spotify search results found")
        return None
    tracks = result['tracks']['items']
    if not tracks:
        logging.error("No tracks found")
        return None
    return tracks


def search_spotify_albums(query: str, limit: int = 2) -> dict | None:
    result = READ_ONLY_SP.search(
        q=query,
        limit=limit,
        type="album",
        market=SEARCH_MARKET)
    if result is None:
        logging.error("No spotify search results found")
        return None
    albums = result['albums']['items']
    if not albums:
        logging.error("No albums found")
        return None
    return albums


def add_spotify_tracks_to_library(track_ids: list[str]):
    # This can accept more than just track_ids!
    AUTHORIZED_SP.current_user_saved_tracks_add(tracks=track_ids)


def remove_spotify_tracks_from_library(track_ids):
    AUTHORIZED_SP.current_user_saved_tracks_delete(tracks=track_ids)


def get_saved_tracks(limit=20) -> list | None:
    user_library_playlist = AUTHORIZED_SP.current_user_saved_tracks(
        limit=limit)
    if not user_library_playlist:
        raise ValueError(
            "No tracks found in user library. Add logic in get_saved_tracks.")
    saved_tracks = user_library_playlist['items']
    return saved_tracks


def extract_track_ids(tracks=None, from_playlist=False):
    if not tracks:
        logging.info("Getting user library tracks")
        tracks = get_saved_tracks()
        from_playlist = True
    if len(tracks) > 100:
        logging.info(f"This list of tracks has more than {len(tracks)} items.")

    track_ids = []
    for idx, track in enumerate(tracks):
        if from_playlist:
            if "track" not in track:
                logging.error(
                    "No track key found in playlist item. Fix the logic in extract_track_ids.")
                raise SystemExit
            track = track['track']
        # print(f"Track {idx+1}. {track['name']} by {track['artists'][0]['name']}")
        track_ids.append(track['id'])
    return track_ids


def extract_album_ids(tracks=None, from_playlist=False, warning_count=30):
    if not tracks:
        logging.info("Getting user library tracks")
        tracks = get_saved_tracks()
        from_playlist = True
    if len(tracks) > warning_count:
        logging.warning(f"This list of tracks has {len(tracks)} items.")

    album_ids = []
    for idx, track in enumerate(tracks):
        if from_playlist:
            track = track["track"]

        # If it's a track, enter the album
        if "album" in track:
            album = track['album']
        else:
            logging.info("Missing album key. Assuming this is an album.")
            album = track
            # logging.error("No track key found in playlist item. Fix the logic in extract_album_ids.")
        album_id = album['id']
        if album_id not in album_ids:
            print(
                f"Album {idx+1}: {album['name']} by {album['artists'][0]['name']}")
            album_ids.append(album_id)
    logging.info(f'Found {len(album_ids)} unique albums')
    return album_ids


def print_items(items, from_playlist=False, limit=10):
    print("Items:")
    print("=======")
    for idx, track in enumerate(items):
        if idx >= limit:
            break
        if from_playlist:
            track = track['track']
        print(f"{idx+1}. {track['name']} by {track['artists'][0]['name']}")
    print()


class TrackSchema(BaseModel):
    name: str
    artist: str
    album: str


def prune_library(
        acceptable_songs: list[str],
        acceptable_artists: list[str],
        prune_limit: int = 99) -> list | None:
    playlist_tracks = get_saved_tracks()
    if not playlist_tracks:
        logging.error("No tracks in your library found")
        return None
    # print("All tracks:")
    # print("=======")
    # print_items(playlist_tracks, from_playlist=True, limit=prune_limit)
    removed_ids = []
    for idx, item in enumerate(playlist_tracks):
        if idx >= prune_limit:
            break
        track_name = item['track']['name']
        artist_name = item['track']['artists'][0]['name']
        album_name = item['track']['album']['name']

        if track_name in acceptable_songs or artist_name in acceptable_artists:
            continue
            # print(f"{idx+1}. {track_name} by {artist_name}")
        else:
            track_id = item['track']['id']
            removed_ids.append(track_id)
            print(f"Removing {track_name} by {artist_name}")
            # Remove from playlist
            playlist_tracks.pop(idx)
    if len(removed_ids) > 20:
        print("Attempting to remove 20+ tracks!")
        if prune_limit != 100:
            logging.warning(
                "prune_limit must be set to 100 as confirmation remove this many songs")
            logging.info("No songs removed")
            return playlist_tracks
    if removed_ids:
        remove_spotify_tracks_from_library(removed_ids)
        print(f"Removed {len(removed_ids)} tracks")
        return playlist_tracks
    return playlist_tracks


def test_functions():
    query = DEFAULT_QUERY
    limit = 1
    ADD_TO_LIB = False
    result_tracks = search_spotify_tracks(query, limit=limit)
    if not result_tracks:
        print("No tracks found")
        return None
    print_items(result_tracks)

    # Add the tracks to the user's library
    if ADD_TO_LIB:
        track_ids = [track['id'] for track in result_tracks]
        print(f"Adding {len(track_ids)} tracks to library")
        add_spotify_tracks_to_library(track_ids)


def res_urls_from_query(
        query: str,
        filter_type="",
        max_urls: int = 1,
        include_substring: str | None = None) -> None | list[str]:
    assert filter_type in ["", "track",
                           "playlist", "album"], "Invalid filter type"
    if filter_type:
        logging.info(f"Restricting search to {filter_type}")
    query = query.strip()
    if not query:
        logging.error("No query provided")
        return None
    # filter_substring = ' "ruth b"' # default
    filter_substring = ""
    if include_substring:
        filter_substring = f' "{include_substring.strip()}"'

    filter_suffix = f" {FILTER_DOMAIN}{filter_type}{filter_substring}"
    formatted_query = f"{query}{filter_suffix}".strip()
    query = formatted_query
    res = perform_lookup(query, max_results=max_urls)
    if not res:
        return None
    res_urls = [str(r["url"]) for r in res]
    return res_urls


def guess_album_name_from_song_name(song_name: str) -> str | None:
    """Uses the album name of the top track found from the query.

    Args:
        song_name (str): This is a search query for a track.

    Returns:
        str | None: The album name of the top track found from the search.
    """
    res = search_spotify_tracks(song_name, limit=1)
    print_items(res)
    if not res:
        print("No results found")
        return None
    album_name = res[0]['album']['name']
    return str(album_name)


def query_to_top_spotify_hits(
        query: str,
        filter_type: str = "track",
        limit=3) -> dict:
    """Get the top Spotify hits for the given query."""
    res_urls = res_urls_from_query(
        query, filter_type=filter_type, max_urls=limit)
    uri_items = uris_from_spotify_urls(res_urls)
    if not uri_items:
        print("No uri items found")
        return None
    result_object = decode_uris(uri_items, expand_tracks=True)
    return result_object


def get_albums_from_result_object(result_object: dict) -> list:
    """Get the album information for all items in the result object."""
    albums = []
    # TODO: Check if this plays nice with all 3 types
    if "playlists" in result_object and result_object["playlists"]:
        for playlist in result_object["playlists"]:
            albums.extend(
                extract_album_ids(
                    playlist["tracks"]["items"],
                    from_playlist=True))
    if "tracks" in result_object and result_object["tracks"]:
        albums.extend(extract_album_ids(result_object["tracks"]))
    if "albums" in result_object and result_object["albums"]:
        albums.extend(extract_album_ids(result_object["albums"]))
    return albums


def wacky_testing():
    # get the song for the query
    query = "birthday gia margaret"
    url_limit = 1
    # spotify_limit = 1
    # res = search_spotify_tracks(query, limit=spotify_limit)
    # print_items(res)
    # if not res:
    #     raise SystemExit
    res_urls = res_urls_from_query(
        query, filter_type="track", max_urls=url_limit)
    if not res_urls:
        logging.error("No urls found")
        raise SystemExit
    uri_items = uris_from_spotify_urls(res_urls)
    if not uri_items:
        logging.error("No uri items found")
        raise SystemExit
    result_object = decode_uris(uri_items, expand_tracks=True)
    if not result_object:
        logging.error("No result object found")
        raise SystemExit
    albums = get_albums_from_result_object(result_object)


def add_top_track_to_library_from_query(query: str):
    tracks = search_spotify_tracks(query, limit=1)
    if not tracks:
        logging.error("No track found")
        return None
    track_id = tracks[0]['id']
    track_ids = [track_id]
    add_spotify_tracks_to_library(track_ids)


def main():
    max_urls = 3
    # Get the spotify URLS that best fit the query
    query = DEFAULT_QUERY
    filter_type = ""
    res_urls = res_urls_from_query(
        query, filter_type=filter_type, max_urls=max_urls)
    if not res_urls:
        print("No results found")
        return None
    # Print the URLs
    logging.info(f"{len(res_urls)} URLs found:")
    for url in res_urls:
        print(url)

    # Get the URIs from the URLs
    uri_items = uris_from_spotify_urls(res_urls)
    if not uri_items:
        print("No uri items found")
        return None

    # Get the track, playlist, and album information
    result_object = decode_uris(uri_items, expand_tracks=True)
    return result_object


if __name__ == "__main__":
    # main()
    test_functions()

### Example payload
# import requests
# def search_spotify(q, result_type, limit, auth_token):
#     assert result_type in ["album", "artist", "playlist", "track", "show", "episode"], "Invalid type"
#     SPOTIFY_AUTH_TOKEN = environ.get("SPOTIFY_AUTH_TOKEN")
#     url = "https://api.spotify.com/v1/search"
#     params = {
#         "q": q,
#         "type": result_type,
#         "market": "US",
#         "limit": limit
#     }
#     headers = {
#         "Authorization": f"Bearer {auth_token}"
#     }
#     response = requests.get(url, params=params, headers=headers)