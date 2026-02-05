"""
intervals_fetch.py — intervals.icu API client for pipeline integration
======================================================================
Foundation module used by sync_athlete_data.py (Phase 1),
fetch_fit_files.py (Phase 2), and GitHub Actions automation (Phase 3).

Usage as library:
    from intervals_fetch import IntervalsClient
    client = IntervalsClient(api_key="...", athlete_id="i224884")
    activities = client.get_activities("2024-01-01", "2024-12-31")
    wellness = client.get_wellness("2024-01-01", "2024-12-31")
    fit_bytes = client.download_fit("i224884:1234567890")

Usage as standalone (quick test):
    python intervals_fetch.py --test

Environment variables (alternative to constructor args):
    INTERVALS_API_KEY=your_key
    INTERVALS_ATHLETE_ID=i12345

    These can be set as system environment variables, or in a .env file
    placed in the same directory as this script. System env vars take
    precedence over .env values.
"""

import os
import sys
import gzip
import json
import time
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing 'requests' — run: pip install requests --break-system-packages")


def _load_dotenv():
    """Load .env file from script directory if it exists (no external dependency)."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            # Don't overwrite existing env vars (system takes precedence)
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()


class IntervalsClient:
    """Client for intervals.icu REST API."""

    BASE_URL = "https://intervals.icu/api/v1"
    
    # Rate limiting: intervals.icu doesn't document limits, but be conservative
    REQUEST_DELAY_S = 0.2          # 200ms between requests
    MAX_RETRIES = 3
    RETRY_BACKOFF_S = 2.0

    def __init__(self, api_key: str = None, athlete_id: str = None):
        self.api_key = api_key or os.environ.get("INTERVALS_API_KEY", "")
        self.athlete_id = athlete_id or os.environ.get("INTERVALS_ATHLETE_ID", "")
        
        if not self.api_key:
            raise ValueError("API key required — pass api_key= or set INTERVALS_API_KEY")
        if not self.athlete_id:
            raise ValueError("Athlete ID required — pass athlete_id= or set INTERVALS_ATHLETE_ID")
        
        self.session = requests.Session()
        self.session.auth = ("API_KEY", self.api_key)
        self.session.headers.update({"Accept": "application/json"})
        self._last_request_time = 0

    def _throttle(self):
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY_S:
            time.sleep(self.REQUEST_DELAY_S - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: dict = None, accept_binary: bool = False) -> requests.Response:
        """GET with retry logic and rate limiting."""
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(self.MAX_RETRIES):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                
                if resp.status_code == 401:
                    raise AuthError("Authentication failed — check API key")
                if resp.status_code == 403:
                    raise AuthError("Access forbidden — check API key permissions")
                if resp.status_code == 404:
                    return None  # Not found is valid for some endpoints
                if resp.status_code == 429:
                    # Rate limited — back off
                    wait = self.RETRY_BACKOFF_S * (attempt + 1) * 2
                    print(f"  Rate limited, waiting {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                    
                resp.raise_for_status()
                return resp
                
            except requests.exceptions.Timeout:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_BACKOFF_S * (attempt + 1))
                    continue
                raise
            except requests.exceptions.ConnectionError:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_BACKOFF_S * (attempt + 1))
                    continue
                raise
        
        raise RuntimeError(f"Failed after {self.MAX_RETRIES} retries: {url}")

    # =========================================================================
    # Activities
    # =========================================================================
    def get_activities(self, oldest: str, newest: str = None) -> list:
        """
        Fetch all activities in date range.
        
        Args:
            oldest: Start date (YYYY-MM-DD)
            newest: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            List of activity dicts. Key fields:
            - id: str (e.g. "i224884:1234567890")
            - start_date_local: str (ISO datetime)
            - type: str ("Run", "Ride", "Workout", etc.)
            - icu_training_load: float (hrTSS)
            - name: str
            - distance: float (meters)
            - moving_time: int (seconds)
            - elapsed_time: int (seconds)
        """
        if newest is None:
            newest = datetime.now().strftime("%Y-%m-%d")
        
        resp = self._get(
            f"/athlete/{self.athlete_id}/activities",
            params={"oldest": oldest, "newest": newest}
        )
        
        if resp is None:
            return []
        return resp.json()

    def get_activity(self, activity_id: str) -> Optional[dict]:
        """Fetch a single activity by ID."""
        resp = self._get(f"/activity/{activity_id}")
        if resp is None:
            return None
        return resp.json()

    # =========================================================================
    # FIT File Download
    # =========================================================================
    def download_fit(self, activity_id: str) -> Optional[bytes]:
        """
        Download FIT file for an activity.
        
        Args:
            activity_id: intervals.icu activity ID
            
        Returns:
            Raw FIT file bytes (decompressed), or None if not available.
        """
        resp = self._get(f"/activity/{activity_id}/file")
        if resp is None:
            return None
        
        content = resp.content
        
        # intervals.icu returns gzip-compressed FIT files
        # Try to decompress; if it fails, assume it's already raw
        try:
            content = gzip.decompress(content)
        except (gzip.BadGzipFile, OSError):
            pass  # Already decompressed or not gzip
        
        return content

    # =========================================================================
    # Wellness (Weight)
    # =========================================================================
    def get_wellness(self, oldest: str, newest: str = None) -> list:
        """
        Fetch wellness records in date range.
        
        Args:
            oldest: Start date (YYYY-MM-DD)
            newest: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            List of wellness dicts. Key fields:
            - id: str (date, e.g. "2024-01-15")
            - weight: float (kg) or None
            - restingHR: int or None
            - hrv: float or None
            - sleepSecs: int or None
        """
        if newest is None:
            newest = datetime.now().strftime("%Y-%m-%d")
        
        resp = self._get(
            f"/athlete/{self.athlete_id}/wellness",
            params={"oldest": oldest, "newest": newest}
        )
        
        if resp is None:
            return []
        return resp.json()

    # =========================================================================
    # Convenience methods
    # =========================================================================
    def get_non_running_activities(self, oldest: str, newest: str = None) -> list:
        """
        Fetch non-running activities with calorie data.
        
        Filters out Run and VirtualRun (handled by pipeline).
        Returns activities that have calories > 0.
        """
        all_acts = self.get_activities(oldest, newest)
        
        running_types = {"Run", "VirtualRun"}
        return [
            act for act in all_acts
            if act.get("type") not in running_types
            and (act.get("calories") or 0) > 0
        ]

    def get_weight_data(self, oldest: str, newest: str = None) -> dict:
        """
        Fetch weight data as {date_str: weight_kg} dict.
        
        Only returns dates with valid weight > 0.
        """
        wellness = self.get_wellness(oldest, newest)
        
        result = {}
        for rec in wellness:
            date_str = rec.get("id", "")
            weight = rec.get("weight")
            if date_str and weight and weight > 0:
                result[date_str] = float(weight)
        
        return result

    def get_new_activities_since(self, since_date: str, sport_type: str = "Run") -> list:
        """
        Fetch activities of a specific type newer than a given date.
        
        Useful for incremental FIT file sync.
        """
        all_acts = self.get_activities(since_date)
        return [
            act for act in all_acts
            if act.get("type") == sport_type
        ]

    def test_connection(self) -> dict:
        """
        Quick connectivity test. Returns summary dict.
        """
        result = {"ok": False, "athlete_id": self.athlete_id}
        
        try:
            # Fetch last 7 days of activities as a smoke test
            week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            today = datetime.now().strftime("%Y-%m-%d")
            
            acts = self.get_activities(week_ago, today)
            wellness = self.get_wellness(week_ago, today)
            
            result["ok"] = True
            result["activities_7d"] = len(acts)
            result["wellness_7d"] = len(wellness)
            
            if acts:
                latest = max(acts, key=lambda a: a.get("start_date_local", ""))
                result["latest_activity"] = {
                    "name": latest.get("name", "?"),
                    "type": latest.get("type", "?"),
                    "date": latest.get("start_date_local", "?")[:10],
                }
            
        except Exception as e:
            result["error"] = str(e)
        
        return result


class AuthError(Exception):
    """Raised for authentication/authorization failures."""
    pass


# =============================================================================
# CLI: quick test mode
# =============================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="intervals.icu API client")
    parser.add_argument("--test", action="store_true", help="Run connectivity test")
    parser.add_argument("--api-key", default=None, help="API key (or set INTERVALS_API_KEY)")
    parser.add_argument("--athlete-id", default=None, help="Athlete ID (or set INTERVALS_ATHLETE_ID)")
    parser.add_argument("--activities", nargs=2, metavar=("OLDEST", "NEWEST"),
                        help="List activities in date range")
    parser.add_argument("--wellness", nargs=2, metavar=("OLDEST", "NEWEST"),
                        help="List wellness records in date range")
    parser.add_argument("--download-fit", metavar="ACTIVITY_ID",
                        help="Download FIT file for activity")
    args = parser.parse_args()
    
    try:
        client = IntervalsClient(api_key=args.api_key, athlete_id=args.athlete_id)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    
    if args.test:
        print("Testing intervals.icu connection...")
        result = client.test_connection()
        for k, v in result.items():
            print(f"  {k}: {v}")
        return 0 if result["ok"] else 1
    
    if args.activities:
        oldest, newest = args.activities
        acts = client.get_activities(oldest, newest)
        print(f"Found {len(acts)} activities ({oldest} -> {newest})")
        for act in acts[:20]:
            tss = act.get("icu_training_load", "")
            tss_str = f" TSS={tss:.0f}" if tss else ""
            print(f"  {act.get('start_date_local', '?')[:10]}  "
                  f"{act.get('type', '?'):15s}  "
                  f"{act.get('name', '?')}"
                  f"{tss_str}")
        if len(acts) > 20:
            print(f"  ... and {len(acts) - 20} more")
        return 0
    
    if args.wellness:
        oldest, newest = args.wellness
        data = client.get_weight_data(oldest, newest)
        print(f"Found {len(data)} days with weight data ({oldest} -> {newest})")
        for date_str in sorted(data.keys())[-10:]:
            print(f"  {date_str}: {data[date_str]:.1f} kg")
        return 0
    
    if args.download_fit:
        activity_id = args.download_fit
        print(f"Downloading FIT file for {activity_id}...")
        fit_bytes = client.download_fit(activity_id)
        if fit_bytes:
            outname = f"{activity_id.replace(':', '_')}.fit"
            with open(outname, "wb") as f:
                f.write(fit_bytes)
            print(f"  Saved: {outname} ({len(fit_bytes):,} bytes)")
        else:
            print("  No FIT file available for this activity")
        return 0
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
