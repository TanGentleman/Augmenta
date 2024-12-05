import logging
from typing import Literal, Union, Optional, List, Dict, Tuple
from pydantic import BaseModel
from .client import SpotifyClient
from langchain_community.tools.tavily_search import TavilySearchResults

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Spotify client
client = SpotifyClient()

# Constants
SEARCH_MARKET = "US"
FILTER_DOMAIN = "site:open.spotify.com/"
ALLOW_AUTHORIZED = True
DEFAULT_QUERY = "2 poor kids"


class Track(BaseModel):
    name: str
    artists: list
    album: str
    id: str
    type: Literal["track"]


class Album(BaseModel):
    name: str
    artists: list
    id: str
    type: Literal["album"]


class Playlist(BaseModel):
    name: str
    id: str
    # This dictionary has a key "items" that is list[PlaylistTrackObjects]
    tracks: dict
    type: Literal["playlist"]


class PlaylistTrackObject(BaseModel):
    name: str
    track: list
    # The list contains Track or Episode objects, filter out the episodes
    type: Literal["track", "episode"]


def perform_lookup(
        query: str, max_results: int = 3) -> list[dict[str, Union[str, int]]] | None:
    """Perform a lookup for the given query."""
    results = TavilySearchResults(max_results=max_results).invoke(query)
    if not results:
        logger.error("No Tavily results found")
        return None
    return results


def check_scope(scopes: list[str] = None) -> bool:
    """Check if the Spotify client is authorized to run the function."""
    
    if not scopes:
        return True
        
    if client is None:
        return False
        
    is_authorized = all(
        [scope in client.scopes for scope in scopes])
    return is_authorized


def id_tuples_from_spotify_urls(urls: list[str]) -> list[tuple[str, str]]:
    """
    Given a list of Spotify URLs, extract the types and IDs."""
    id_tuples = []
    for url in urls:
        if not url.startswith("https://open.spotify.com/"):
            logger.error("Invalid URL")
            continue
        id_type = url.split("/")[3]
        spot_id = url.split("/")[-1].split("?")[0]
        if id_type not in ["track", "playlist", "album"]:
            logger.info(f"Ignoring URL of type {id_type}")
            continue
        id_tuples.append((id_type, spot_id))
    if not id_tuples:
        logger.error("No valid objects found")
        return []
    return id_tuples


def decode_id_tuples(id_tuples: List[Tuple[str, str]], expand_tracks=False, limit=10):
    """Given a list of tuples with type and ID, decode the tracks, playlists, and albums."""
    results = {"tracks": [], "playlists": [], "albums": [], "id_tuples": []}
    
    for id_type, spot_id in id_tuples:
        if (id_type, spot_id) in results["id_tuples"]:
            logger.info(f"Skipping duplicate {id_type}")
            continue
            
        if id_type == "playlist":
            playlist = client.get_playlist(spot_id)
            if not playlist:
                continue
            print("Playlist:", playlist["name"])
            if expand_tracks:
                playlist_tracks = playlist["tracks"]["items"]
                if len(playlist_tracks) > limit:
                    logger.info(f"{len(playlist_tracks)} tracks truncated to {limit}.")
                print_items(playlist_tracks, from_playlist=True, limit=limit)
            results["playlists"].append(playlist)
            
        elif id_type == "track":
            track = client.get_track(spot_id)
            if not track:
                continue
            print(f"Track: {track['name']} by {track['artists'][0]['name']}")
            results["tracks"].append(track)
            
        elif id_type == "album":
            album = client.get_album(spot_id)
            if not album:
                continue
            print("Album:", album["name"])
            if expand_tracks and album.get("tracks", {}).get("items"):
                tracks = album["tracks"]["items"]
                if len(tracks) > limit:
                    logger.info(f"{len(tracks)} tracks truncated to {limit}.")
                print_items(tracks, from_playlist=False, limit=limit)
            results["albums"].append(album)
            
        results["id_tuples"].append((id_type, spot_id))
        
    if not any(results.values()):
        logger.error("No valid items found")
        return None
    return results


def search_spotify(query: str, result_type: str = "track", limit: int = 2) -> list | dict[str] | None:
    """Search Spotify for tracks or albums."""
    return client.search(query, result_type, limit)


def add_spotify_tracks_to_library(track_values: list[str]) -> Optional[bool]:
    """Add tracks to the user's Spotify library."""
    return client.add_tracks_to_library(track_values)


def remove_spotify_tracks_from_library(track_values: list[str], bulk=False) -> Optional[bool]:
    """Remove tracks from the user's Spotify library."""
    return client.remove_tracks_from_library(track_values, bulk)


def create_playlist(playlist_name: str, public=False, description: str = "") -> Optional[Dict]:
    """Create a new playlist for the user."""
    return client.create_playlist(playlist_name, public, description)


def get_user_library_playlist(limit=20) -> Optional[List[Dict]]:
    """Get tracks from user's library."""
    return client.get_user_library(limit)


def extract_item_ids(items=False, from_playlist=False):
    if items is False:
        logger.info("Getting user library tracks")
        tracks = get_user_library_playlist()
        from_playlist = True
    if len(items) > 100:
        logger.info(f"This list of tracks has more than {len(tracks)} items.")
    item_ids = []
    seen_ids = set()
    for item in items:
        if from_playlist:
            track_id = item['track']['id']
        else:
            track_id = item['id']
        if track_id in seen_ids:
            logger.info(f"Skipping duplicate track {track_id}")
            continue
        item_ids.append(track_id)
        seen_ids.add(track_id)
    return item_ids


def print_album_names(items=False, from_playlist=False, warning_count=30):
    if items is None:
        raise ValueError("None value for items passed to get_album_names.")
    if items is False:
        logger.info("Getting user library tracks")
        playlist_items = get_user_library_playlist()
        items = playlist_items
        from_playlist = True
    if len(items) > warning_count:
        logger.warning(f"This list of items has {len(items)} items.")

    album_names = []
    item_ids = []
    seen_ids = set()
    for item in items:
        if from_playlist:
            # This is a list of PlaylistTrackObjects
            album = item['track']['album']
        else:
            # This is an AlbumObject
            if item["type"] == "track":
                album = item['album']
            elif item["type"] == "album":
                album = item
            else:
                logger.error(f"Skipping {item['type']}")
                continue
        assert album["type"] == "album", "Invalid album type"
        album_name = album["name"]
        album_id = album["id"]
        if album_id in seen_ids:
            logger.info(f"Skipping duplicate album {album_id}")
            continue
        album_names.append(album_name)
        item_ids.append(album_id)
        seen_ids.add(album_id)
    logger.info(f'Found {len(item_ids)} unique albums')
    print("Album names:")
    print("============")
    for idx, album_name in enumerate(album_names):
        print(f"{idx+1}. {album_name}")
    return item_ids


def print_items(items, from_playlist=False, limit=10):
    print("Items:")
    print("=======")
    for idx, item in enumerate(items):
        if idx >= limit:
            break
        if from_playlist:
            playlist_track = item['track']
            if playlist_track["type"] != "track":
                assert playlist_track["type"] == "episode"
                logger.info("Skipping episode")
                continue
            item_name = playlist_track['name']
            item_type = playlist_track['type']
        else:
            item_name = item['name']
            item_type = item['type']
        print(f"{idx+1}. ({item_type}): {item_name}")
    print()


def prune_library(
    acceptable_songs: list[str],
    acceptable_artists: list[str],
    playlist_tracks: list | None = None,  # If not provided, get from user library
        bulk=False) -> list | None:
    """Prune the user's library based on the acceptable songs and artists."""
    # TODO: Make this robust, input should be dict with allowed/forbidden
    # songs/artists/albums/ids
    if playlist_tracks is None:
        playlist_tracks = get_user_library_playlist()
    if not playlist_tracks:
        logger.error("No tracks in your library found")
        return None
    # print("All tracks:")
    # print("=======")
    # print_items(playlist_tracks, from_playlist=True, limit=prune_limit)
    prune_limit = 500 if bulk else 20
    removed_ids = []
    for idx, item in enumerate(playlist_tracks):
        if idx >= prune_limit:
            logger.info(f"Pruning limit reached at {prune_limit}")
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
            logger.info(f"Removing {track_name} by {artist_name}")
            # Remove from playlist
            playlist_tracks.pop(idx)
            # TODO: Confirm that this is safe and working
    if removed_ids:
        if len(removed_ids) > 20 and not bulk:
            logger.warning(
                "Bulk removal not enabled. Set bulk=True to remove all tracks.")
            logger.info(
                "No songs removed from library. Returning modified playlist.")
            return playlist_tracks
        remove_spotify_tracks_from_library(removed_ids, bulk=bulk)
        logger.info(f"Removed {len(removed_ids)} tracks")
    return playlist_tracks


def start_playback(context_uri: str = "") -> Optional[bool]:
    """Start playback on the user's Spotify account."""
    track = client.search("kurt hugo listen to your heart", result_type="track", limit=1)
    if not track:
        logger.error("No track found")
        return None
    uri = track[0]['uri']
    uri_values = [uri]
    return client.start_playback(uris=uri_values)


def test_functions():
    """Test basic search functionality."""
    query = DEFAULT_QUERY
    limit = 1
    ADD_TO_LIB = False
    result_tracks = client.search(query, result_type="track", limit=limit)
    if not result_tracks:
        logger.info("No tracks found")
        return None
    print_items(result_tracks)

    # Add the tracks to the user's library
    if ADD_TO_LIB:
        track_ids = [track['id'] for track in result_tracks]
        logger.info(f"Adding {len(track_ids)} tracks to library")
        client.add_tracks_to_library(track_ids)


def query_tavily_get_urls(
        query: str,
        filter_type: Literal["", "track", "playlist", "album"] = "",
        max_urls: int = 1,
        include_substring: str | None = None) -> None | list[str]:
    if filter_type not in ["", "track", "playlist", "album"]:
        logger.error("Invalid filter type")
        raise ValueError("Invalid filter type")
    if filter_type:
        logger.info(f"Restricting search to {filter_type}")
    query = query.strip()
    if not query:
        logger.error("No query provided")
        raise ValueError("No query provided")
    # filter_substring = ' "ruth b"' # default
    filter_substring = f' "{include_substring.strip()}"' if include_substring else ""
    filter_suffix = f" {FILTER_DOMAIN}{filter_type}{filter_substring}"
    # This should already be stripped() right?
    formatted_query = f"{query}{filter_suffix}"
    query = formatted_query
    res = perform_lookup(query, max_results=max_urls)
    if not res:
        return []
    res_urls = [str(r["url"]) for r in res]
    return res_urls


def guess_album_name_from_song_name(song_name: str) -> str | None:
    """Uses the album name of the top track found from the query."""
    tracks = client.search(query=song_name, result_type="track", limit=1)
    if not tracks:
        logger.error("No results found")
        return None
    logger.info(f"Input song name: {tracks[0]['name']}")
    album_name = tracks[0]['album']['name']
    assert isinstance(album_name, str), "Album name must be a string"
    return album_name


def query_tavily_return_spotify_dict(query: str, filter_type: str = "track", limit=3) -> dict:
    """Get the top Spotify hits for the given query."""
    res_urls = query_tavily_get_urls(query, filter_type=filter_type, max_urls=limit)
    if not res_urls:
        logger.error("No tavily results found")
        return {}
    id_tuples = id_tuples_from_spotify_urls(res_urls)
    if not id_tuples:
        logger.error("No valid spotify items found for query")
        return {}
    spotify_dict = decode_id_tuples(id_tuples, expand_tracks=True)
    if not spotify_dict:
        logger.error("No result object found")
        return {}
    return spotify_dict


def extract_all_albums_from_spotify_dict(spotify_dict: dict) -> list[Album]:
    """Get the album information for all items in the result object."""
    albums = []
    # TODO: Check if this plays nice with all 3 types
    if "playlists" in spotify_dict and spotify_dict["playlists"]:
        for playlist in spotify_dict["playlists"]:
            playlist_tracks = playlist["tracks"]["items"]
            for track in playlist_tracks:
                if track["type"] == "episode":
                    logger.info("Skipping episode")
                    continue
                album = track["track"]["album"]
                Album(**album)
                albums.append(album)
    if "tracks" in spotify_dict and spotify_dict["tracks"]:
        for track in spotify_dict["tracks"]:
            album = track["album"]
            Album(**album)
            if album in albums:
                logger.info("Skipping duplicate album")
                # Do I really need to report dupes?
                continue
            albums.append(album)
    if "albums" in spotify_dict and spotify_dict["albums"]:
        for album in spotify_dict["albums"]:
            Album(**album)
            albums.append(album)

    if not albums:
        logger.error("No albums found in result object")
        return []
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
    res_urls = query_tavily_get_urls(
        query, filter_type="track", max_urls=url_limit)
    if not res_urls:
        logger.error("No urls found")
        raise SystemExit
    id_tuples = id_tuples_from_spotify_urls(res_urls)
    if not id_tuples:
        logger.error("No id tuples found")
        raise SystemExit
    spotify_dict = decode_id_tuples(id_tuples, expand_tracks=True)
    if not spotify_dict:
        logger.error("No result object found")
        raise SystemExit
    albums = extract_all_albums_from_spotify_dict(spotify_dict)
    if not albums:
        logger.error("No albums found")
        raise SystemExit
    print_album_names(items=albums, from_playlist=False)


def add_track_from_query(query: str):
    """Add a track to the user's library from a query."""
    tracks = search_spotify(query, result_type="track", limit=1)
    if not tracks:
        logger.error("No track found")
        return None
    track_id = tracks[0]['id']
    track_ids = [track_id]
    add_spotify_tracks_to_library(track_ids)
    return None


def print_my_library(limit=20):
    print_items(
        get_user_library_playlist(
            limit=limit),
        from_playlist=True,
        limit=limit)


def test_guess() -> bool:
    """Test album name guessing functionality."""
    song_name = "every breath you take"
    try:
        album_name = guess_album_name_from_song_name(song_name)
        logger.info(f"Album name: {album_name}")
        return len(album_name) > 0
    except Exception as e:
        logger.error(e)
        return False


def test_albums() -> bool:
    """Test album retrieval functionality."""
    query = "every breath you take"
    spotify_dict = query_tavily_return_spotify_dict(query)
    if not spotify_dict:
        logger.error("No result object found")
        return False
    albums = extract_all_albums_from_spotify_dict(spotify_dict)
    if not albums:
        logger.error("No albums found")
        return False
    return True


def test_prune():
    """Test library pruning functionality."""
    acceptable_songs = ["Every Breath You Take", "Roxanne"]
    acceptable_artists = ["The Police"]
    prune_library(acceptable_songs, acceptable_artists)


def test_workflow():
    """Test the complete workflow."""
    query = "songs your neighbors listen to. Stargazing songs by The Neighbours"
    USE_TAVILY = False
    
    if USE_TAVILY:
        res_urls = query_tavily_get_urls(query, filter_type="playlist")
        if not res_urls:
            logger.error("No result urls found")
            return None
        id_tuples = id_tuples_from_spotify_urls(res_urls)
        if not id_tuples:
            logger.error("No id tuples found")
            return None
        spotify_dict = decode_id_tuples(id_tuples, expand_tracks=True)
        if not spotify_dict:
            logger.error("No result object found")
            return None
        playlists = spotify_dict["playlists"]
        playlist_tracks = playlists[0]["tracks"]["items"]
        print_items(playlist_tracks, from_playlist=True)
    else:
        playlists = client.search(query, result_type="playlist", limit=1)
        
    if not playlists:
        logger.error("No playlists found")
        return None
        
    logger.info("Using the first playlist")
    playlist = playlists[0]
    assert playlist["type"] == "playlist", "Invalid type"
    logger.info(f"Playlist: {playlist['name']}")
    
    # Add to library
    uri_list = [playlist["uri"]]
    logger.info(f"Adding tracks: {uri_list} to library")
    client.add_tracks_to_library(uri_list)
    
    playlist_tracks = playlist["tracks"]["items"]
    print_items(playlist_tracks, from_playlist=True)
    item_ids = extract_item_ids(playlist_tracks, from_playlist=True)
    
    with open("temp.txt", "w") as f:
        f.write(str(item_ids))

def download_playlist_from_url(url: str, limit=30) -> None:
    """Download a playlist from a Spotify URL and add tracks to library.
    
    Args:
        url: Spotify playlist URL
        limit: Maximum number of tracks to process
    """
    id_tuples = id_tuples_from_spotify_urls([url])
    if not id_tuples:
        logger.error("No id tuples found")
        return None
        
    spotify_dict = decode_id_tuples(id_tuples, expand_tracks=True, limit=limit)
    if not spotify_dict:
        logger.error("No result object found")
        return None
        
    playlists = spotify_dict["playlists"]
    if not playlists:
        logger.error("No playlists found")
        return None
        
    playlist_tracks = playlists[0]["tracks"]["items"]
    track_ids = extract_item_ids(playlist_tracks, from_playlist=True)
    
    # Save track IDs for reference
    with open("temp.txt", "w") as f:
        f.write(str(track_ids))
        logger.info("Saved track IDs to temp.txt")
        
    # Add tracks to library
    client.add_tracks_to_library(track_ids)

def test_playlist_dl():
    """Test playlist download functionality."""
    input_url = input("Enter the Spotify URL: ")
    assert input_url.startswith("https://open.spotify.com/"), "Invalid URL"
    download_playlist_from_url(input_url)


def run_tests():
    """Run all tests."""
    assert test_albums(), "Test 2 failed"

def main():
    """Main entry point."""
    max_urls = 3
    query = DEFAULT_QUERY
    filter_type = ""
    
    res_urls = query_tavily_get_urls(query, filter_type=filter_type, max_urls=max_urls)
    if not res_urls:
        logger.info("No results found")
        return None
        
    logger.info(f"{len(res_urls)} URLs found:")
    for url in res_urls:
        print(url)

    id_tuples = id_tuples_from_spotify_urls(res_urls)
    if not id_tuples:
        logger.error("No id tuples found")
        return None

    spotify_dict = decode_id_tuples(id_tuples, expand_tracks=True)
    return spotify_dict




if __name__ == "__main__":
    run_tests()
    # test_prune()
    # main()
    # test_functions()
    # tracks = add_track_from_query("every breath you take")
    # if tracks:
    #     print_items(tracks, from_playlist=True)
    # print_my_library()
    # audio_analysis_from_query("dandelions ruth")


# Example payload
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
