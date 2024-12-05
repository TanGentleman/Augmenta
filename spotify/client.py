import logging
import os
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth


class SpotifyClient:
    """Client for interacting with the Spotify API."""

    def __init__(self, allow_authorized: bool = True):
        """Initialize the Spotify client.

        Args:
            allow_authorized (bool): Whether to enable authorized operations. Defaults to True.
        """
        self._configure_logging()
        self._configure_valid_types()
        self._configure_constants()
        self._configure_api(allow_authorized)

    def _configure_logging(self) -> None:
        """Configure logging settings."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _configure_api(self, allow_authorized: bool) -> None:
        """Configure API clients.
        
        Args:
            allow_authorized: Whether to initialize authorized client
        """
        # Load environment variables
        load_dotenv()
        self._client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self._client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

        if not self._client_id or not self._client_secret:
            raise ValueError("Spotify client ID and secret not found in environment variables")

        # Initialize read-only client
        self._read_only_client = Spotify(auth_manager=SpotifyClientCredentials(
            client_id=self._client_id,
            client_secret=self._client_secret
        ))

        # Initialize authorized client if allowed
        self._authorized_client = None
        if allow_authorized:
            scope_string = " ".join(self.AUTHORIZED_CLIENT_SCOPES)
            self._authorized_client = Spotify(auth_manager=SpotifyOAuth(
                client_id=self._client_id,
                client_secret=self._client_secret,
                redirect_uri="http://localhost:3000",
                scope=scope_string
            ))

    def _configure_valid_types(self) -> None:
        """Configure valid search and result types."""
        self._valid_search_types = frozenset({"track", "album", "playlist", "both"})
        self._valid_result_types = frozenset({"track", "album", "playlist"})

    def _configure_constants(self) -> None:
        """Configure API constants."""
        self.SEARCH_MARKET = "US"
        self.FILTER_DOMAIN = "site:open.spotify.com/"
        
        # Scopes
        self.READ_LIBRARY_SCOPE = "user-library-read"
        self.MODIFY_LIBRARY_SCOPE = "user-library-modify"
        self.MODIFY_PLAYLIST_SCOPE = "playlist-modify-private"
        self.MODIFY_PLAYBACK_SCOPE = "user-modify-playback-state"
        self.READ_PLAYBACK_SCOPE = "user-read-playback-state"

        self.AUTHORIZED_CLIENT_SCOPES = [
            self.READ_LIBRARY_SCOPE,
            self.MODIFY_LIBRARY_SCOPE,
            self.MODIFY_PLAYLIST_SCOPE,
            self.MODIFY_PLAYBACK_SCOPE,
            self.READ_PLAYBACK_SCOPE
        ]

    def check_scope(self, scopes: Optional[List[str]] = None) -> bool:
        """Check if client is authorized for given scopes."""
        if scopes:
            if self._authorized_client is None:
                return False
            return all(scope in self.AUTHORIZED_CLIENT_SCOPES for scope in scopes)
        return True

    def search(
        self,
        query: str,
        result_type: str = "track",
        limit: int = 2
    ) -> Optional[Union[List, Dict[str, List]]]:
        """Search Spotify for content.
        
        Args:
            query: Search query string
            result_type: Type of content to search for ("track", "album", "playlist", "both")
            limit: Maximum number of results to return
            
        Returns:
            Optional[Union[List, Dict[str, List]]]: Search results
        """
        if result_type not in self._valid_search_types:
            raise ValueError(f"Invalid result_type. Must be one of: {self._valid_search_types}")

        if result_type == "both":
            result_type = "track,album"

        self.logger.info(f"Searching for {result_type}s")
        
        result = self._read_only_client.search(
            q=query,
            limit=limit,
            type=result_type,
            market=self.SEARCH_MARKET
        )

        return self._process_search_results(result, result_type)

    def _process_search_results(
        self,
        result: Dict,
        result_type: str
    ) -> Optional[Union[List, Dict[str, List]]]:
        """Process and validate search results.
        
        Args:
            result: Raw search results
            result_type: Type of content searched for
            
        Returns:
            Optional[Union[List, Dict[str, List]]]: Processed results
        """
        if not result:
            self.logger.error("No results found")
            return None

        if result_type == "track,album":
            return self._process_combined_results(result)
        
        items = result.get(f"{result_type}s", {}).get('items', [])
        if not items:
            self.logger.error(f"No {result_type}s found")
            return None

        self.logger.info(f"Found {len(items)} {result_type}s")
        return items

    def _process_combined_results(self, result: Dict) -> Optional[Dict[str, List]]:
        """Process combined track and album search results.
        
        Args:
            result: Raw search results containing both tracks and albums
            
        Returns:
            Optional[Dict[str, List]]: Processed results with tracks and albums
        """
        tracks = result.get('tracks', {}).get('items', [])
        albums = result.get('albums', {}).get('items', [])
        
        if not (tracks or albums):
            self.logger.error("No tracks or albums found")
            return None
            
        return {
            "tracks": tracks,
            "albums": albums,
            "playlists": []
        }

    # Library Management Methods
    def add_tracks_to_library(self, track_values: List[str]) -> Optional[bool]:
        """Add tracks to user's Spotify library.
        
        Args:
            track_values: List of track URIs, URLs or IDs
            
        Returns:
            Optional[bool]: Success status
        """
        if not self.check_scope([self.MODIFY_LIBRARY_SCOPE]):
            self.logger.error("Not authorized to modify library")
            return None
            
        return self._authorized_client.current_user_saved_tracks_add(tracks=track_values)

    def remove_tracks_from_library(
        self,
        track_values: List[str],
        bulk: bool = False
    ) -> Optional[bool]:
        """Remove tracks from user's Spotify library.
        
        Args:
            track_values: List of track URIs, URLs or IDs
            bulk: Whether to allow bulk removal (>50 tracks)
            
        Returns:
            Optional[bool]: Success status
        """
        if not self.check_scope([self.MODIFY_LIBRARY_SCOPE]):
            self.logger.error("Not authorized to modify library")
            return None

        if len(track_values) > 50 and not bulk:
            self.logger.error("Bulk removal not enabled. Set bulk=True to remove >50 tracks")
            return None

        return self._authorized_client.current_user_saved_tracks_delete(tracks=track_values)

    def get_user_library(self, limit: int = 20) -> Optional[List[Dict]]:
        """Get tracks from user's library.
        
        Args:
            limit: Maximum number of tracks to return
            
        Returns:
            Optional[List[Dict]]: List of saved tracks
        """
        if not self.check_scope([self.READ_LIBRARY_SCOPE]):
            self.logger.error("Not authorized to read library")
            return None

        result = self._authorized_client.current_user_saved_tracks(limit=limit)
        return result.get('items', []) if result else None

    # Playlist Management Methods
    def create_playlist(
        self,
        name: str,
        public: bool = False,
        description: str = ""
    ) -> Optional[Dict]:
        """Create a new playlist.
        
        Args:
            name: Playlist name
            public: Whether playlist should be public
            description: Playlist description
            
        Returns:
            Optional[Dict]: Created playlist details
        """
        if not self.check_scope([self.MODIFY_PLAYLIST_SCOPE]):
            self.logger.error("Not authorized to create playlists")
            return None

        user_id = self._authorized_client.current_user()["id"]
        return self._authorized_client.user_playlist_create(
            user=user_id,
            name=name,
            public=public,
            description=description
        )

    def add_tracks_to_playlist(
        self,
        playlist_id: str,
        track_uris: List[str]
    ) -> Optional[Dict]:
        """Add tracks to a playlist.
        
        Args:
            playlist_id: Playlist ID
            track_uris: List of track URIs to add
            
        Returns:
            Optional[Dict]: Response with snapshot ID
        """
        if not self.check_scope([self.MODIFY_PLAYLIST_SCOPE]):
            self.logger.error("Not authorized to modify playlists")
            return None

        return self._authorized_client.playlist_add_items(
            playlist_id=playlist_id,
            items=track_uris
        )

    def remove_tracks_from_playlist(
        self,
        playlist_id: str,
        track_uris: List[str]
    ) -> Optional[Dict]:
        """Remove tracks from a playlist.
        
        Args:
            playlist_id: Playlist ID
            track_uris: List of track URIs to remove
            
        Returns:
            Optional[Dict]: Response with snapshot ID
        """
        if not self.check_scope([self.MODIFY_PLAYLIST_SCOPE]):
            self.logger.error("Not authorized to modify playlists")
            return None

        return self._authorized_client.playlist_remove_all_occurrences_of_items(
            playlist_id=playlist_id,
            items=track_uris
        )

    # Playback Control Methods
    def start_playback(
        self,
        device_id: Optional[str] = None,
        context_uri: Optional[str] = None,
        uris: Optional[List[str]] = None,
        offset: Optional[Dict] = None,
        position_ms: Optional[int] = None
    ) -> Optional[bool]:
        """Start or resume playback.
        
        Args:
            device_id: ID of device to play on
            context_uri: Spotify URI of context to play
            uris: List of track URIs to play
            offset: Offset into context
            position_ms: Position in track (milliseconds)
            
        Returns:
            Optional[bool]: Success status
        """
        if not self.check_scope([self.MODIFY_PLAYBACK_SCOPE]):
            self.logger.error("Not authorized to control playback")
            return None

        return self._authorized_client.start_playback(
            device_id=device_id,
            context_uri=context_uri,
            uris=uris,
            offset=offset,
            position_ms=position_ms
        )

    def pause_playback(self, device_id: Optional[str] = None) -> Optional[bool]:
        """Pause playback.
        
        Args:
            device_id: ID of device to pause
            
        Returns:
            Optional[bool]: Success status
        """
        if not self.check_scope([self.MODIFY_PLAYBACK_SCOPE]):
            self.logger.error("Not authorized to control playback")
            return None

        return self._authorized_client.pause_playback(device_id=device_id)

    def next_track(self, device_id: Optional[str] = None) -> Optional[bool]:
        """Skip to next track.
        
        Args:
            device_id: ID of device to control
            
        Returns:
            Optional[bool]: Success status
        """
        if not self.check_scope([self.MODIFY_PLAYBACK_SCOPE]):
            self.logger.error("Not authorized to control playback")
            return None

        return self._authorized_client.next_track(device_id=device_id)

    def previous_track(self, device_id: Optional[str] = None) -> Optional[bool]:
        """Skip to previous track.
        
        Args:
            device_id: ID of device to control
            
        Returns:
            Optional[bool]: Success status
        """
        if not self.check_scope([self.MODIFY_PLAYBACK_SCOPE]):
            self.logger.error("Not authorized to control playback")
            return None

        return self._authorized_client.previous_track(device_id=device_id)

    # Device Methods
    def get_available_devices(self) -> Optional[List[Dict]]:
        """Get user's available devices.
        
        Returns:
            Optional[List[Dict]]: List of available devices
        """
        if not self.check_scope([self.READ_PLAYBACK_SCOPE]):
            self.logger.error("Not authorized to read playback state")
            return None

        result = self._authorized_client.devices()
        return result.get('devices', []) if result else None

    def get_current_playback(self) -> Optional[Dict]:
        """Get information about current playback.
        
        Returns:
            Optional[Dict]: Current playback information
        """
        if not self.check_scope([self.READ_PLAYBACK_SCOPE]):
            self.logger.error("Not authorized to read playback state")
            return None

        return self._authorized_client.current_playback()

    # Utility Methods
    def get_track(self, track_id: str) -> Optional[Dict]:
        """Get track details.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Optional[Dict]: Track details
        """
        try:
            return self._read_only_client.track(track_id)
        except Exception as e:
            self.logger.error(f"Failed to get track: {e}")
            return None

    def get_album(self, album_id: str) -> Optional[Dict]:
        """Get album details.
        
        Args:
            album_id: Spotify album ID
            
        Returns:
            Optional[Dict]: Album details
        """
        try:
            return self._read_only_client.album(album_id)
        except Exception as e:
            self.logger.error(f"Failed to get album: {e}")
            return None

    def get_playlist(self, playlist_id: str) -> Optional[Dict]:
        """Get playlist details.
        
        Args:
            playlist_id: Spotify playlist ID
            
        Returns:
            Optional[Dict]: Playlist details
        """
        try:
            return self._read_only_client.playlist(playlist_id)
        except Exception as e:
            self.logger.error(f"Failed to get playlist: {e}")
            return None


if __name__ == "__main__":
    with SpotifyClient() as client:
        # Example usage
        results = client.search("The Beatles", result_type="track", limit=1)
        if results:
            track = results[0]
            print(f"Found track: {track['name']} by {track['artists'][0]['name']}")