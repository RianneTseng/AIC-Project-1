import os
import time
import random
import yt_dlp
import ffmpeg
import isodate
from googleapiclient.discovery import build

# YouTube API Key
API_KEY = ""
OUTPUT_DIR = "youtube_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Accent types & search keywords
ACCENTS = ["American", "British"]
SEARCH_KEYWORDS = [
    "English pronunciation", 
    "English news", 
    "English speech training",
    "English storytelling", 
    "English language podcast", 
    "TED English talk",
    "BBC English report", 
    "CNN English interview", 
    "Daily English conversation",
    "Advanced English vocabulary"
]

# Download parameters
MAX_RETRIES = 3  # Number of retries for downloading
NUM_VIDEOS = 100  # Minimum number of videos to download per search
MAX_DURATION = 600  # Maximum video length (seconds)
MIN_DURATION = 90   # Minimum video length (seconds)
CLIP_COUNT = 5  # Number of clips to extract from each video
CLIP_LENGTH = 15  # Length of each clip (seconds)

# Convert YouTube ISO 8601 format duration to seconds.
def parse_duration(duration):
    return int(isodate.parse_duration(duration).total_seconds())

# Search YouTube videos and filter out those exceeding max_duration or below min_duration.
def search_youtube(query, max_results=NUM_VIDEOS, max_duration=MAX_DURATION):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    request = youtube.search().list(q=query, part="id,snippet", maxResults=max_results, type="video")
    response = request.execute()

    video_urls = []
    for item in response.get("items", []):
        video_id = item["id"]["videoId"]
        video_details = youtube.videos().list(id=video_id, part="contentDetails").execute()
        duration = video_details["items"][0]["contentDetails"]["duration"]

        seconds = parse_duration(duration)
        if MIN_DURATION <= seconds <= max_duration:
            video_urls.append((f"https://www.youtube.com/watch?v={video_id}", seconds))

    return video_urls

# Download video and skip if HLS error occurs.
def download_with_retries(video_url, output_path):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path.replace(".wav", ""),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}],
        "noplaylist": True,
        "hls_prefer_native": True,
        "ignoreerrors": False,
        "skip_download": False,
        "postprocessor_args": ["-af", "aresample=async=1:min_hard_comp=0.100:first_pts=0"],
        "progress_hooks": [lambda d: stop_on_hls_error(d)]
    }
    
    def stop_on_hls_error(d):
        if d["status"] == "error" and "hls" in d.get("message", "").lower():
            raise Exception("HLS error detected, skipping video")

    start_time = time.time()
    for attempt in range(MAX_RETRIES):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            elapsed_time = time.time() - start_time
            return True, elapsed_time
        except Exception as e:
            print(f"Error downloading {video_url} (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if "HLS error" in str(e):
                return False, 0  # Immediately skip the video
            time.sleep(5)
    return False, 0

# Extract audio clips ensuring non-overlapping segments and sufficient video length.
def extract_audio_clips(audio_path, output_prefix, duration):
    if duration < (CLIP_COUNT * CLIP_LENGTH):
        print(f"Video {audio_path} is too short ({duration}s), skipping and redownloading...")
        return False  # Video too short, needs re-downloading
    
    start_times = set()
    
    for _ in range(CLIP_COUNT):
        while True:
            start_time = random.randint(0, duration - CLIP_LENGTH)
            
            # Ensure new clip is at least 15 seconds apart from existing clips
            if all(abs(start_time - s) >= CLIP_LENGTH for s in start_times):
                start_times.add(start_time)
                break
        
        output_clip = f"{output_prefix}_clip{len(start_times)}.wav"
        (
            ffmpeg.input(audio_path, ss=start_time, t=CLIP_LENGTH)
            .output(output_clip, format='wav')
            .run(overwrite_output=True)
        )
    
    return True  # Successfully extracted audio clips

def main():
    total_downloads = 0
    total_time = 0
    for accent in ACCENTS:
        downloaded_videos = 0
        
        while downloaded_videos < NUM_VIDEOS:
            for query in SEARCH_KEYWORDS:
                search_query = f"{accent} {query}"
                print(f"Searching: {search_query}")
                video_data = search_youtube(search_query)

                for i, (video_url, duration) in enumerate(video_data):
                    if downloaded_videos >= NUM_VIDEOS:
                        break
                    
                    output_audio = os.path.join(OUTPUT_DIR, f"{accent.lower()}_{downloaded_videos}.wav")
                    success, elapsed_time = download_with_retries(video_url, output_audio)
                    
                    if success:
                        if not extract_audio_clips(output_audio, output_audio.replace(".wav", ""), duration):
                            continue  # Skip video if extraction fails (too short), re-download
                        downloaded_videos += 1
                        total_downloads += 1
                        total_time += elapsed_time
                        
                        avg_time_per_video = total_time / total_downloads if total_downloads else 0
                        remaining_videos = (len(ACCENTS) * NUM_VIDEOS) - total_downloads
                        estimated_time_left = avg_time_per_video * remaining_videos
                        
                        print(f"Downloaded {downloaded_videos}/{NUM_VIDEOS} for {accent}. Remaining time: {estimated_time_left:.2f} sec")

if __name__ == "__main__":
    main()
    print("\nAll downloads completed successfully!")
