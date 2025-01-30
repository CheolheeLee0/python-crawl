import yt_dlp
import os
import threading
from queue import Queue
import time

# Define constants
NUM_THREADS = 5
MAX_RETRIES = 3
RETRY_DELAY = 5

def download_video(queue, channel_name, stop_event):
    while not stop_event.is_set():
        try:
            video_info = queue.get(timeout=1)
            if video_info is None:
                break
            
            url = f"https://www.youtube.com/watch?v={video_info['id']}"
            title = video_info.get('title', 'Unknown Title').replace('/', '_')
            filename = f"{title}"
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'outtmpl': f'downloads/{channel_name}/{filename}',
                'progress_hooks': [lambda d: stop_event.is_set() and yt_dlp.utils.bug_reports_message()],
            }
            
            for attempt in range(MAX_RETRIES):
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])
                    print(f"Successfully downloaded: {title}")
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        print(f"Attempt {attempt + 1} failed for {title}: {str(e)}. Retrying...")
                        time.sleep(RETRY_DELAY)
                    else:
                        print(f"Failed to download {title} after {MAX_RETRIES} attempts: {str(e)}")
        except Queue.Empty:
            continue
        finally:
            queue.task_done()

def crawl_and_download_channel_videos(channel_url, stop_event):
    ydl_opts = {
        'extract_flat': 'in_playlist',
        'ignoreerrors': True,
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            result = ydl.extract_info(channel_url, download=False)
            
            if 'entries' in result:
                videos = result['entries']
                # Change how channel_name is generated
                channel_name = channel_url.split('/')[-2].replace('@', '')
                print(f"Found {len(videos)} videos in channel: {channel_name}. Starting download...")
                
                # Create channel-specific directory if it doesn't exist
                channel_dir = os.path.join('downloads', channel_name)
                if not os.path.exists(channel_dir):
                    os.makedirs(channel_dir)
                
                # Create a queue to hold video information
                video_queue = Queue()
                
                # Create and start worker threads
                threads = []
                for _ in range(NUM_THREADS):
                    t = threading.Thread(target=download_video, args=(video_queue, channel_name, stop_event))
                    t.daemon = True
                    t.start()
                    threads.append(t)
                
                # Add video information to the queue
                for video in videos:
                    if video:
                        video_queue.put(video)
                
                # Add None to the queue to signal the threads to exit
                for _ in range(NUM_THREADS):
                    video_queue.put(None)
                
                # Wait for all tasks to be completed
                video_queue.join()
                
                # Wait for all threads to finish
                for t in threads:
                    t.join()
                
                print(f"All downloads completed for channel: {channel_name}")
            else:
                print("No videos found in the channel.")
        except Exception as e:
            print(f"An error occurred while crawling the channel: {str(e)}")

def main():
    channels = [
        "@YCombinator",
    ]

    stop_event = threading.Event()
    
    try:
        for channel_url in channels:
            crawl_and_download_channel_videos(f"https://www.youtube.com/{channel_url}/videos", stop_event)
    except KeyboardInterrupt:
        print("Stopping downloads...")
        stop_event.set()
    
    print("All downloads completed or stopped.")

if __name__ == "__main__":
    main()