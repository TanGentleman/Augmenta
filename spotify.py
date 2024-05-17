from typing import Union
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()
from os import environ
SPOTIFY_CLIENT_ID = environ.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = environ.get("SPOTIFY_CLIENT_SECRET")

from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth

from langchain_community.tools.tavily_search import TavilySearchResults

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
#     return response.json()

ALLOW_AUTHORIZED = True
if ALLOW_AUTHORIZED:
    print("Allowing authorized Spotify API. Make sure your key is up to date.")

READ_ONLY_SP = Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID,
                                                            client_secret=SPOTIFY_CLIENT_SECRET))

if ALLOW_AUTHORIZED:
    AUTHORIZED_SP = Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIFY_CLIENT_ID,
                                                    client_secret=SPOTIFY_CLIENT_SECRET,
                                                    redirect_uri="http://localhost:3000",
                                                    scope="user-library-read user-library-modify"))
else:
    AUTHORIZED_SP = None

QUERY = "2 poor kids"

def perform_lookup(query: str, max_results:int = 3) -> list[dict[str, Union[str, int]]]:
    """Perform a lookup for the given query."""
    results = TavilySearchResults(max_results=max_results).invoke(query)
    if not results:
        raise ValueError("No results found")
    return results

def decode_spotify_urls(urls: str, expand_playlist = False, limit = 10):
    """
    Given a list of Spotify URLs, decode the URIs and print the tracks, playlists, and albums.

    Args:
        urls (str): A list of Spotify URLs.
        expand_playlist (bool, optional): If True, print tracks in the playlist. Defaults to False.
        limit (int, optional): The limit of tracks to show. Defaults to 10.

    Returns:
        dict: A dictionary containing the tracks, playlists, and albums.
    """
    results = {}
    results["tracks"] = []
    results["playlists"] = []
    results["albums"] = []
    for url in urls:
        assert url.startswith("https://open.spotify.com/"), "Invalid URL"
        uri_type = url.split("/")[3]
        uri = url.split("/")[-1].split("?")[0]
        if uri_type == "playlist":
            # TODO: Get playlist name
            playlist_URI = uri
            playlist = READ_ONLY_SP.playlist(playlist_URI)
            playlist_name = playlist["name"]
            print("Playlist:", playlist_name)
            if expand_playlist:
                # Use playlist object to get tracks
                tracks = playlist["tracks"]["items"]
                if len(tracks) > limit:
                    print(f"Will only show first {limit} tracks")
                print_playlist_items(tracks, limit=limit)
                # playlist_track_names = [x["track"]["name"] for x in tracks]
                # print(playlist_track_names)
            results["playlists"].append(playlist)
            
        elif uri_type == "track":
            track_URI = uri
            track = READ_ONLY_SP.track(track_URI)
            track_name = track["name"]
            artist_name = track["artists"][0]["name"]
            print(f"Track: {track_name} by {artist_name}")
            results["tracks"].append(track)
        elif uri_type == "album":
            album_URI = uri
            album = READ_ONLY_SP.album(album_URI)
            album_name = album["name"]
            print("Album:", album_name)
            results["albums"].append(album)
        else:
            print("Ignoring URL")
    if not results:
        print("No valid objects found")
        return None
    return results
    
def search_spotify_tracks(query: str, limit: int = 2):
    result = READ_ONLY_SP.search(q=query, limit=limit, type="track")
    if "tracks" not in result:
        print("No tracks found")
        return None
    tracks = result['tracks']['items']
    if not tracks:
        print("No tracks found")
        return None
    return tracks

def search_spotify_albums(query: str, limit:int = 2):
    result = READ_ONLY_SP.search(q=query, limit=limit, type="album", market="US")
    if "albums" not in result:
        print("No albums found")
        return None
    albums = result['albums']['items']
    if not albums:
        print("No albums found")
        return None
    return albums

def add_spotify_tracks_to_library(track_ids: list[str]):
    # This can accept more than just track_ids!
    AUTHORIZED_SP.current_user_saved_tracks_add(tracks=track_ids)

def remove_spotify_tracks_from_library(track_ids):
    AUTHORIZED_SP.current_user_saved_tracks_delete(tracks=track_ids)

def get_saved_tracks():
    user_library_playlist = AUTHORIZED_SP.current_user_saved_tracks()
    saved_tracks = user_library_playlist['items']
    return saved_tracks

def get_track_ids_from_tracks(tracks = None, from_playlist = False):
    if not tracks:
        print("Getting user library tracks")
        tracks = get_saved_tracks()
        from_playlist = True
    if len(tracks) > 100:
        print(f"This list of tracks has more than {len(tracks)} items.")
    if from_playlist:
        track_ids = [track['track']['id'] for track in tracks]
    else:
        track_ids = [track['id'] for track in tracks]
    return track_ids

def get_album_ids_from_playlist():
    saved_tracks = get_saved_tracks()
    unique_album_ids = []
    # album_ids = [track['track']['album']['id'] for track in saved_tracks]

    for track in saved_tracks:
        album_id = track['track']['album']['id']
        if album_id not in unique_album_ids:
            unique_album_ids.append(album_id)
        return unique_album_ids
    else:
        print("No album IDs found")
        return None

def print_items(items):
    print("Items:")
    print("=======")
    for idx, track in enumerate(items):
        print(f"{idx+1}. {track['name']} by {track['artists'][0]['name']}")

def print_playlist_items(tracks, limit=10):
    print("Saved tracks:")
    print("=======")
    for idx, item in enumerate(tracks):
        track = item['track']
        if idx >= limit:
            break
        print(f"{idx+1}. {track['name']} by {track['artists'][0]['name']}")


class TrackSchema(BaseModel):
    name: str
    artist: str
    album: str

def test_functions():
    query = QUERY
    limit = 1
    iteration_limit = 99
    ADD_TO_LIB = True
    FROM_SAVED = False
    if FROM_SAVED:
        playlist_tracks = get_saved_tracks()
        if not playlist_tracks:
            print("No tracks in your library found")
            return None
        print_playlist_items(playlist_tracks, limit)
    else:
        result_tracks = search_spotify_tracks(query, limit=limit)
        print_items(result_tracks)
        if not result_tracks:
            print("No tracks found")
            return None
    
    # Add the tracks to the user's library
    if ADD_TO_LIB and not FROM_SAVED:
        track_ids = [track['id'] for track in result_tracks]
        print(f"Adding {len(track_ids)} tracks to library")
        add_spotify_tracks_to_library(track_ids)
    
    # Print the user library
    playlist_tracks = get_saved_tracks()
    print_playlist_items(playlist_tracks, limit=10)

    prune_collection = True
    acceptable_songs = "Good Luck"
    acceptable_artists = ["Ruth B.", "Tennis", "Rick and Morty"]
    final_tracks = []
    if prune_collection:
        print("Final tracks:")
        print("=======")
        removed_ids = []
        for idx, item in enumerate(playlist_tracks):
            if idx >= iteration_limit:
                break
            track_name = item['track']['name']
            artist_name = item['track']['artists'][0]['name']
            album_name = item['track']['album']['name']

            if track_name in acceptable_songs or artist_name in acceptable_artists:
                print(f"{idx+1}. {track_name} by {artist_name}")
            else:
                track_id = item['track']['id']
                removed_ids.append(track_id)
                print(f"Removing {track_name} by {artist_name}")
        if removed_ids:
            remove_spotify_tracks_from_library(removed_ids)
            print(f"Removed {len(removed_ids)} tracks")
            final_tracks.append(TrackSchema(name=track_name, artist=artist_name, album=album_name).model_dump())
    
    return final_tracks


def res_urls_from_query(query: str, filter_type = "", max_urls: int = 1):
    assert filter_type in ["", "track", "playlist", "album"], "Invalid filter type"
    if filter_type:
        print(f"Restricting search to {filter_type}")
    suffix = f"site:open.spotify.com/{filter_type}"
    formatted_query = f"{query} {suffix}".strip()
    query = formatted_query
    res = perform_lookup(query, max_results=max_urls)
    res_urls = [r["url"] for r in res]
    if not res_urls:
        print("No URLs found")
        return None
    return res_urls

def get_spotify_uris_from_urls(urls: list[str]):
    track_uris = []
    playlist_uris = []
    album_uris = []
    for url in urls:
        uri = url.split("/")[-1].split("?")[0]
        if url.startswith("https://open.spotify.com/track/"):
            track_uris.append(uri)
        elif url.startswith("https://open.spotify.com/playlist/"):
            playlist_uris.append(uri)
        elif url.startswith("https://open.spotify.com/album/"):
            album_uris.append(uri)
        else:
            print("Ignoring URL")

    if not any([track_uris, playlist_uris, album_uris]):
        print("No valid URIs found")
        return None
    return track_uris, playlist_uris, album_uris
    
def print_info_from_spotify_uris(track_uris, playlist_uris, album_uris, expand_albums = False):
    if track_uris:
        print("Tracks:")
        tracks = READ_ONLY_SP.tracks(track_uris)
        track_names = [track["name"] for track in tracks["tracks"]]
        print(track_names)
    if playlist_uris:
        print("Only showing first playlist:")
        # only do first playlist
        playlist_uri = playlist_uris[0]
        playlists = READ_ONLY_SP.playlist(playlist_uri)
        playlist_name = playlists["name"]
        print("playlist name:", playlist_name)
    if album_uris:
        print("Albums:")
        albums = READ_ONLY_SP.albums(album_uris)
        if expand_albums:
            for album in albums["albums"]:
                print('Tracks in', album["name"], 'by', album["artists"][0]["name"])
                print("=======")
                for track in album["tracks"]["items"]:
                    print(track['name'])
                    # print(f"{track['name']} by {track['artists'][0]['name']}")
                print()
        else:
            album_names = [album["name"] for album in albums["albums"]]
            print(album_names)

def main():
    max_urls = 3
    ### Get the spotify URLS that best fit the query
    query = "ruth b dandelions"
    filter_type = "album"
    res_urls = res_urls_from_query(query, filter_type=filter_type, max_urls=max_urls)

    # Print the URLs
    for i in res_urls:
        print(i)  

    ### Get the URIs from the URLs
    track_uris, playlist_uris, album_uris = get_spotify_uris_from_urls(res_urls)

    if not any([track_uris, playlist_uris, album_uris]):
        print("No valid URIs found")
        raise SystemExit
    
    ### Get the track, playlist, and album information          
    print_info_from_spotify_uris(track_uris, playlist_uris, album_uris, expand_albums=True)

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

def query_to_top_spotify_hits(query: str, filter_type: str = "track", limit = 3) -> dict:
    """Get the top Spotify hits for the given query."""
    res_urls = res_urls_from_query(query, filter_type=filter_type, max_urls=limit)
    track_uris, playlist_uris, album_uris = get_spotify_uris_from_urls(res_urls)
    print_info_from_spotify_uris(track_uris, playlist_uris, album_uris, expand_albums=False)
    results = search_spotify_tracks(query, limit=1)
    if not results:
        raise ValueError("No results found")
    return results[0]

if __name__ == "__main__":
    # test_functions()
    # get the song for the query
    query = "i believe i can fly"
    url_limit = 3
    res = search_spotify_tracks(query, limit=1)
    # print_items(res)
    res_urls = res_urls_from_query(query, filter_type="playlist", max_urls=url_limit)
    # decode the urls directly
    decode_spotify_urls(res_urls, expand_playlist=True)

    # or get the track, playlist, and album information
    track_uris, playlist_uris, album_uris = get_spotify_uris_from_urls(res_urls)
    print("VERSUS:\n\n")
    print_info_from_spotify_uris(track_uris, playlist_uris, album_uris, expand_albums=False)
