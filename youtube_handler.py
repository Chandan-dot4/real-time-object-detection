"""
YouTube Video Stream Handler using yt-dlp.
Resolves direct video stream URLs, extracts metadata, manages streaming formats,
and handles network exceptions or YouTube bot restrictions with local fallbacks.
"""

import os
import re
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yt_dlp
import config

logger = logging.getLogger("YouTubeHandler")
logging.basicConfig(level=logging.INFO)

class YouTubeStreamHandler:
    def __init__(self):
        # yt-dlp configuration optimized for low latency and smooth cv2.VideoCapture streaming
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'noplaylist': True,
            'socket_timeout': 15,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
            }
        }
        self.stream_cache = {}

    @staticmethod
    def is_valid_youtube_url(url: str) -> bool:
        """Validates whether the provided string is a plausible YouTube video link."""
        if not url or not isinstance(url, str):
            return False
        clean_url = url.strip()
        pattern = r'^(https?://)?(www\.)?(youtube\.com|youtu\.be|m\.youtube\.com)/.+$'
        return bool(re.match(pattern, clean_url))

    def extract_direct_stream_url(self, youtube_url: str) -> Dict[str, Any]:
        """
        Uses yt-dlp to extract the direct MP4/HLS streaming URL and video details.
        """
        youtube_url = youtube_url.strip()
        if not self.is_valid_youtube_url(youtube_url):
            return {
                "success": False,
                "stream_url": None,
                "title": "",
                "duration": 0,
                "thumbnail": "",
                "resolution": "",
                "error": "Invalid YouTube URL format. Please provide a standard youtube.com or youtu.be link."
            }

        if youtube_url in self.stream_cache:
            cached_data = self.stream_cache[youtube_url]
            logger.info("Returning cached stream for: %s", cached_data.get("title"))
            return cached_data

        try:
            logger.info("Resolving YouTube stream via yt-dlp: %s", youtube_url)
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                if not info:
                    raise ValueError("Failed to extract video information from YouTube.")

                if 'entries' in info and len(info['entries']) > 0:
                    info = info['entries'][0]

                stream_url = None
                width = 1280
                height = 720

                # Inspect available formats
                raw_formats = info.get('formats', [])
                valid_formats = [f for f in raw_formats if f.get('url') and f.get('vcodec') != 'none']

                if valid_formats:
                    # Select progressive or HLS stream targeting 720p or 480p or 360p for real-time OpenCV processing
                    chosen_format = None
                    for target_h in [720, 480, 360, 1080]:
                        matches = [f for f in valid_formats if f.get('height') == target_h and f.get('ext') == 'mp4']
                        if matches:
                            chosen_format = matches[0]
                            break

                    if not chosen_format:
                        for target_h in [720, 480, 360, 1080]:
                            matches = [f for f in valid_formats if f.get('height') == target_h]
                            if matches:
                                chosen_format = matches[0]
                                break

                    if not chosen_format:
                        chosen_format = valid_formats[-1]

                    stream_url = chosen_format['url']
                    width = chosen_format.get('width', 1280) or 1280
                    height = chosen_format.get('height', 720) or 720
                elif info.get('url'):
                    stream_url = info['url']
                    width = info.get('width', 1280) or 1280
                    height = info.get('height', 720) or 720

                if not stream_url:
                    raise ValueError("No compatible video stream URL found for OpenCV.")

                title = info.get('title', 'YouTube Traffic Stream')
                duration = info.get('duration', 0)
                thumbnail = info.get('thumbnail', '')

                result = {
                    "success": True,
                    "stream_url": stream_url,
                    "title": title,
                    "duration": duration,
                    "thumbnail": thumbnail,
                    "resolution": f"{width}x{height}",
                    "error": None
                }
                self.stream_cache[youtube_url] = result
                logger.info("Successfully extracted stream for: %s (%dx%d)", title, width, height)
                return result

        except Exception as e:
            err_msg = str(e)
            logger.error("yt-dlp stream extraction error: %s", err_msg)
            hint = "Please check your network or try a local video file."
            if "Sign in to confirm you’re not a bot" in err_msg or "HTTP Error 429" in err_msg:
                hint = "YouTube rate-limited automated stream extraction from this IP. Please use the Local Video or Fallback Sample option."
            elif "Private video" in err_msg or "Video unavailable" in err_msg:
                hint = "The video is private, restricted, or unavailable."

            return {
                "success": False,
                "stream_url": None,
                "title": "",
                "duration": 0,
                "thumbnail": "",
                "resolution": "",
                "error": f"{err_msg[:160]}... {hint}"
            }

youtube_handler = YouTubeStreamHandler()
