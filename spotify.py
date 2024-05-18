import logging
from typing import Literal, Union, Optional
from dotenv import dotenv_values
from pydantic import BaseModel
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
env_config = dotenv_values()

SPOTIFY_CLIENT_ID = env_config.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = env_config.get("SPOTIFY_CLIENT_SECRET")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SEARCH_MARKET = "US"
FILTER_DOMAIN = "site:open.spotify.com/"
ALLOW_AUTHORIZED = True
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

# Schemas
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
    tracks: dict # This dictionary has a key "items" that is list[PlaylistTrackObjects]
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

def check_scope() -> bool:
    """Check if the Spotify client is authorized to read/modify the library."""
    is_authorized = AUTHORIZED_SP is not None
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


def decode_id_tuples(id_tuples: str, expand_tracks=False, limit=10):
    """
    Given a list of tuples with type and ID, decode the tracks, playlists, and albums.

    Args:
        id_tuples (list[Tuple[str, str]]): A list of tuples containing the type and ID.
        expand_tracks (bool, optional): Whether to expand the tracks. Defaults to False.
        limit (int, optional): The limit for the number of tracks to print. Defaults to 10.
    Returns:
        dict: A dictionary containing the tracks, playlists, albums, and id_tuples.
    """
    results = {"tracks": [], "playlists": [], "albums": [], "id_tuples": []}
    for id_type, spot_id in id_tuples:
        # Check for duplicate
        if (id_type, spot_id) in results["id_tuples"]:
            logger.info(f"Skipping duplicate {id_type}")
            continue
        if id_type not in ["track", "playlist", "album"]:
            logger.info(f"Ignoring type {id_type}")
            continue
        if id_type == "playlist":
            playlist = READ_ONLY_SP.playlist(spot_id)
            # Handle null playlist here?
            if playlist.get("type") != "playlist":
                logger.error("Unexpected type")
                raise ValueError("Invalid playlist type")
            playlist_name = playlist["name"]
            print("Playlist:", playlist_name)
            if expand_tracks:
                playlist_tracks = playlist["tracks"]["items"]
                if len(playlist_tracks) > limit:
                    logger.info(f"{len(playlist_tracks)} tracks truncated to {limit}.")
                # Use playlist object to get tracks
                print_items(playlist_tracks, from_playlist=True, limit=limit)
            results["playlists"].append(playlist)
        elif id_type == "track":
            track = READ_ONLY_SP.track(spot_id)
            # Handle null track here?
            if track.get("type") != "track":
                logger.warning("Skipping unexpected type")
                continue
            track_name = track["name"]
            artist_name = track["artists"][0]["name"] # Is this safe?
            print(f"Track: {track_name} by {artist_name}")
            results["tracks"].append(track)
        elif id_type == "album":
            album = READ_ONLY_SP.album(spot_id)
            # Handle null album here?
            if album.get("type") != "album":
                logger.error("Unexpected type")
                raise ValueError("Invalid album type")
            album_name = album["name"]
            if expand_tracks:
                tracks = album["tracks"].get("items", [])
                if not tracks:
                    logger.error("No tracks found in album. Not saving")
                    continue
                # These are apparently SimplifiedTrackObjects
                if len(tracks) > limit:
                    logger.info(f"{len(tracks)} tracks truncated to {limit}.")
                print("Album:", album_name, "Tracks:")
                print_items(tracks, from_playlist=False, limit=limit)
            else:
                print("Album:", album_name)
            results["albums"].append(album)
        else:
            raise ValueError("Invalid id_type")
        # Add the id_tuple to the results
        results["id_tuples"].append((id_type, spot_id))
    if not any(results.values()):
        logger.error("No valid items found")
        return None
    return results


def search_spotify(query: str, result_type: str = "track", limit: int = 2) -> list | dict[str] | None:
    """Search Spotify for tracks or albums."""
    if result_type not in ["track", "album", "both"]:
        logger.error("Only track and album searches are supported")
        raise ValueError("Invalid result type")
    if result_type == "both":
        result_type = "track,album"
    logger.info(f"Searching for {result_type}s")
    result = READ_ONLY_SP.search(q=query, limit=limit, type=result_type, market=SEARCH_MARKET)
    if result_type == "track":
        result = result.get('tracks', {})
    elif result_type == "album":
        result = result.get('albums', {})
    else:
        assert result_type == "track,album", "Invalid result type"
        tracks = result.get('tracks', {}).get('items', [])
        albums = result.get('albums', {}).get('items', [])
        if tracks and albums:
            logger.warning("Both tracks and albums found. Returning response_object.")
            response_object = {"tracks": tracks, "albums": albums, "playlists": []}
            return response_object
        else:
            if tracks:
                logger.info(f"Found {len(tracks)} tracks")
                result = tracks
            elif albums:
                logger.info(f"Found {len(albums)} albums")
                result = albums
            else:
                logger.error("No tracks or albums found")
                return None
    items = result.get('items', [])
    if not items:
        logger.error(f"No {result_type}s found")
        return None
    logger.info(f"Found {len(items)} {result_type}s")
    return items


def add_spotify_tracks_to_library(track_values: list[str]):
    """Add tracks to the user's Spotify library."""
    # track_values is a list of track URIs, URLs or IDs
    is_authorized = check_scope()
    if not is_authorized:
        logger.error("Spotify client not authorized for this function.")
        raise SystemExit
    return AUTHORIZED_SP.current_user_saved_tracks_add(tracks=track_values)


def remove_spotify_tracks_from_library(track_values: list[str], bulk=False):
    """Remove tracks from the user's Spotify library."""
    # track_values is a list of track URIs, URLs or IDs
    is_authorized = check_scope()
    if not is_authorized:
        logger.error("Spotify client not authorized for this function.")
        raise SystemExit
    if len(track_values) > 50:
        logger.warning("This will check against more than 50 tracks.")
        if not bulk:
            logger.error("bulk arg in remove_spotify_tracks_from_library is False. Set to True to remove override.")
            logger.info("No tracks removed.")
            return None
    return AUTHORIZED_SP.current_user_saved_tracks_delete(tracks=track_values)


def get_user_library_playlist(limit=20) -> list | None:
    user_library_playlist = AUTHORIZED_SP.current_user_saved_tracks(
        limit=limit)
    if not user_library_playlist:
        raise ValueError(
            "No tracks found in user library. Add logic in get_user_library_playlist.")
    return user_library_playlist.get('items', [])


def extract_item_ids(items = False, from_playlist=False):
    if not items:
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


def print_album_names(items = False, from_playlist=False, warning_count=30):
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
       bulk = False) -> list | None:
    """Prune the user's library based on the acceptable songs and artists."""
    # TODO: Make this robust, input should be dict with allowed/forbidden songs/artists/albums/ids
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
            logger.warning("Bulk removal not enabled. Set bulk=True to remove all tracks.")
            logger.info("No songs removed from library. Returning modified playlist.")
            return playlist_tracks
        remove_spotify_tracks_from_library(removed_ids, bulk=bulk)
        logger.info(f"Removed {len(removed_ids)} tracks")
    return playlist_tracks


def test_functions():
    query = DEFAULT_QUERY
    limit = 1
    ADD_TO_LIB = False
    result_tracks = search_spotify(query, result_type="track", limit=limit)
    if not result_tracks:
        logger.info("No tracks found")
        return None
    print_items(result_tracks)

    # Add the tracks to the user's library
    if ADD_TO_LIB:
        track_ids = [track['id'] for track in result_tracks]
        logger.info(f"Adding {len(track_ids)} tracks to library")
        add_spotify_tracks_to_library(track_ids)


def res_urls_from_query(
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
    formatted_query = f"{query}{filter_suffix}" # This should already be stripped() right?
    query = formatted_query
    res = perform_lookup(query, max_results=max_urls)
    if not res:
        return []
    res_urls = [str(r["url"]) for r in res]
    return res_urls


def guess_album_name_from_song_name(song_name: str) -> str | None:
    """Uses the album name of the top track found from the query.

    Args:
        song_name (str): This is a search query for a track.

    Returns:
        str | None: The album name of the top track found from the search.
    """
    tracks = search_spotify(query=song_name, result_type="track", limit=1)
    if not tracks:
        logger.error("No results found")
        return None
    logger.info(f"Input song name: {tracks[0]['name']}")
    album_name = tracks[0]['album']['name']
    # print(f"Album name: {album_name}")
    assert isinstance(album_name, str), "Album name must be a string"
    return album_name


def query_tavily_return_spotify_dict(
        query: str,
        filter_type: str = "track",
        limit=3) -> dict:
    """Get the top Spotify hits for the given query."""
    res_urls = res_urls_from_query(
        query, filter_type=filter_type, max_urls=limit)
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
    res_urls = res_urls_from_query(
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
    print_items(get_user_library_playlist(limit=limit), from_playlist=True, limit=limit)

def main():
    max_urls = 3
    # Get the spotify URLS that best fit the query
    query = DEFAULT_QUERY
    filter_type = ""
    res_urls = res_urls_from_query(
        query, filter_type=filter_type, max_urls=max_urls)
    if not res_urls:
        logger.info("No results found")
        return None
    # Print the URLs
    logger.info(f"{len(res_urls)} URLs found:")
    for url in res_urls:
        print(url)

    # Get the id_tuples from the URLs
    id_tuples = id_tuples_from_spotify_urls(res_urls)
    if not id_tuples:
        logger.error("No id tuples found")
        return None

    # Get the track, playlist, and album information
    spotify_dict = decode_id_tuples(id_tuples, expand_tracks=True)
    return spotify_dict

def test_guess():
    song_name = "every breath you take"
    try:
        album_name = guess_album_name_from_song_name(song_name)
        logger.info(f"Album name: {album_name}")
        return len(album_name) > 0
    except Exception as e:
        logger.error(e)
        return False
    
def test_albums():
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
    acceptable_songs = ["Every Breath You Take", "Roxanne"]
    acceptable_artists = ["The Police"]
    prune_library(acceptable_songs, acceptable_artists)

def run_tests():
    # Test 1 - Spotify search track + parse name
    # assert test_guess(), "Test 1 failed"
    # Test 2 
    assert test_albums(), "Test 2 failed"



if __name__ == "__main__":
    run_tests()
    # main()
    # test_functions()
    # tracks = add_track_from_query("every breath you take")
    # if tracks:
    #     print_items(tracks, from_playlist=True)
    # print_my_library()


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