# 변액보험 재생목록 동영상 to mp3

# 필요한 라이브러리들을 임포트
import yt_dlp            # YouTube 비디오 다운로드를 위한 라이브러리
import os               # 운영 체제와 상호작용하기 위한 라이브러리 (파일 및 디렉토리 조작)
import threading        # 멀티스레딩을 위한 라이브러리
from queue import Queue # 스레드 간 작업을 분배하기 위한 큐 클래스
import time            # 시간 관련 기능을 위한 라이브러리
import logging         # 로그 기록을 위한 라이브러리

# 로깅 설정
logging.basicConfig(
    filename='server.log',          # 로그를 기록할 파일 이름
    level=logging.INFO,             # 로깅 레벨을 INFO로 설정 (정보 메시지 기록)
    format='%(asctime)s - %(levelname)s - %(message)s',  # 로그 메시지의 형식
    datefmt='%Y-%m-%d %H:%M:%S'    # 로그에 기록될 날짜 및 시간 형식
)

# 프로그램에서 사용할 상수 정의
NUM_THREADS = 5        # 동시에 실행할 다운로드 스레드의 수
MAX_RETRIES = 3        # 다운로드 실패 시 최대 재시도 횟수
RETRY_DELAY = 5        # 재시도 전 대기할 시간 (초)
ONLY_FETCH_INFO = False  # 실제 다운로드 모드로 변경

def download_video(queue, channel_name, stop_event):
    """
    비디오 다운로드를 처리하는 함수
    queue: 다운로드할 비디오 정보가 담긴 큐
    channel_name: 채널 또는 재생목록의 이름
    stop_event: 다운로드 중단을 위한 이벤트 객체
    """
    while not stop_event.is_set():  # 중단 신호가 설정되지 않은 동안 계속 실행
        try:
            # 큐에서 비디오 정보를 가져옴 (1초 타임아웃)
            video_info = queue.get(timeout=1)
            if video_info is None:   # 종료 신호를 받으면 루프 종료
                break
            
            # 비디오 URL과 제목 생성
            url = f"https://www.youtube.com/watch?v={video_info['id']}"  # 비디오 URL 생성
            title = video_info.get('title', 'Unknown Title').replace('/', '_')  # 제목 가져오기 (슬래시 대체)
            filename = f"{title}"  # 파일 이름 설정
            
            logging.info(f"Starting download: {title}")  # 다운로드 시작 로그 기록
            
            # yt-dlp 다운로드 옵션 설정
            ydl_opts = {
                'format': 'bestaudio/best',              # 최고 품질의 오디오 형식 선택
                'postprocessors': [{                     # 다운로드 후 처리 설정
                    'key': 'FFmpegExtractAudio',        # FFmpeg를 사용하여 오디오 추출
                    'preferredcodec': 'wav',            # WAV 형식으로 변환
                }],
                'outtmpl': f'downloads/{channel_name}/{filename}',  # 저장할 경로 및 파일 이름 설정
                'progress_hooks': [lambda d: stop_event.is_set() and yt_dlp.utils.bug_reports_message()],  # 진행 상황 후크
            }
            
            # 최대 재시도 횟수만큼 다운로드 시도
            for attempt in range(MAX_RETRIES):
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # yt-dlp 객체 생성
                        ydl.download([url])  # 비디오 다운로드
                    logging.info(f"Successfully downloaded: {title}")  # 다운로드 성공 로그 기록
                    break  # 다운로드 성공 시 루프 종료
                except Exception as e:  # 다운로드 중 예외 발생 시
                    if attempt < MAX_RETRIES - 1:  # 마지막 시도가 아닐 경우
                        logging.warning(f"Attempt {attempt + 1} failed for {title}: {str(e)}. Retrying...")  # 경고 로그 기록
                        time.sleep(RETRY_DELAY)  # 재시도 전 대기
                    else:  # 마지막 시도에서 실패한 경우
                        logging.error(f"Failed to download {title} after {MAX_RETRIES} attempts: {str(e)}")  # 오류 로그 기록
        except Queue.Empty:  # 큐가 비어 있을 경우
            continue  # 다음 반복으로 넘어감
        finally:
            queue.task_done()  # 큐의 작업 완료 표시

def crawl_and_download_channel_videos(channel_url, stop_event):
    """
    채널 비디오를 크롤링하고 다운로드하는 함수
    channel_url: 다운로드할 채널의 URL
    stop_event: 다운로드 중단을 위한 이벤트 객체
    """
    # yt-dlp 옵션 설정
    ydl_opts = {
        'extract_flat': 'in_playlist',  # 재생목록에서 비디오 정보를 추출
        'ignoreerrors': True,            # 오류 무시
        'quiet': True,                   # 조용한 모드 (출력 최소화)
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # yt-dlp 객체 생성
        try:
            result = ydl.extract_info(channel_url, download=False)  # 채널 정보 추출 (다운로드 하지 않음)
            
            if 'entries' in result:  # 비디오 항목이 있는 경우
                videos = result['entries']  # 비디오 목록 가져오기
                channel_name = channel_url.split('/')[-2].replace('@', '')  # 채널 이름 추출
                logging.info(f"Found {len(videos)} videos in channel: {channel_name}")  # 비디오 수 로그 기록
                
                # CSV 파일에 제목과 링크 저장
                with open(f'{channel_name}_videos.csv', 'w', encoding='utf-8') as csv_file:
                    csv_file.write('Title,Link\n')  # 헤더 작성
                    for video in videos:  # 비디오 목록을 순회
                        title = video.get('title', 'Unknown Title')
                        link = f"https://www.youtube.com/watch?v={video['id']}"
                        csv_file.write(f'{title},{link}\n')  # 제목과 링크 기록
                
                # 채널 전용 디렉토리 생성 (존재하지 않을 경우)
                channel_dir = os.path.join('downloads', channel_name)  # 채널 디렉토리 경로
                if not os.path.exists(channel_dir):  # 디렉토리가 존재하지 않으면
                    os.makedirs(channel_dir)  # 디렉토리 생성
                
                # 비디오 정보를 담을 큐 생성
                video_queue = Queue()  # 큐 객체 생성
                
                # 작업 스레드 생성 및 시작
                threads = []  # 스레드 목록
                for _ in range(NUM_THREADS):  # 지정된 수만큼 스레드 생성
                    t = threading.Thread(target=download_video, args=(video_queue, channel_name, stop_event))  # 스레드 생성
                    t.daemon = True  # 데몬 스레드로 설정 (메인 프로그램 종료 시 함께 종료)
                    t.start()  # 스레드 시작
                    threads.append(t)  # 스레드 목록에 추가
                
                # 비디오 정보를 큐에 추가
                for video in videos:  # 비디오 목록을 순회
                    if video:  # 비디오 정보가 있는 경우
                        video_queue.put(video)  # 큐에 비디오 정보 추가
                
                # 스레드 종료 신호를 큐에 추가
                for _ in range(NUM_THREADS):  # 각 스레드에 대해 종료 신호 추가
                    video_queue.put(None)  # None을 큐에 추가하여 스레드 종료 신호 전달
                
                # 모든 작업이 완료될 때까지 대기
                video_queue.join()  # 큐의 모든 작업이 완료될 때까지 대기
                
                # 모든 스레드가 종료될 때까지 대기
                for t in threads:  # 스레드 목록을 순회
                    t.join()  # 각 스레드가 종료될 때까지 대기
                
                logging.info(f"All downloads completed for channel: {channel_name}")  # 모든 다운로드 완료 로그 기록
            else:
                logging.warning("No videos found in the channel.")  # 비디오가 없는 경우 경고 로그 기록
        except Exception as e:  # 예외 발생 시
            logging.error(f"An error occurred while crawling the channel: {str(e)}")  # 오류 로그 기록

def crawl_and_download_playlist_videos(playlist_url, stop_event):
    """
    재생목록 비디오를 크롤링하고 다운로드하는 함수
    playlist_url: 다운로드할 재생목록의 URL
    stop_event: 다운로드 중단을 위한 이벤트 객체
    """
    # yt-dlp 옵션 설정
    ydl_opts = {
        'extract_flat': 'in_playlist',  # 재생목록에서 비디오 정보를 추출
        'ignoreerrors': True,            # 오류 무시
        'quiet': True,                   # 조용한 모드 (출력 최소화)
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # yt-dlp 객체 생성
        try:
            result = ydl.extract_info(playlist_url, download=False)  # 재생목록 정보 추출 (다운로드 하지 않음)
            
            if ONLY_FETCH_INFO:  # 정보만 가져오는 모드일 경우
                playlist_name = result.get('title', 'Unknown Playlist').replace('/', '_')  # 재생목록 이름 추출
                videos_count = len(result.get('entries', []))  # 비디오 수 계산
                logging.info(f"Playlist: {playlist_name}, Number of videos: {videos_count}")  # 재생목록 이름과 비디오 수 로그 기록
                
                # 결과를 lecture.csv에 저장
                with open('lecture.csv', 'w', encoding='utf-8') as f:  # lecture.csv 파일 열기
                    f.write('Playlist Name,Number of Videos\n')  # 헤더 작성
                    f.write(f'{playlist_name},{videos_count}\n')  # 재생목록 이름과 비디오 수 기록
                
                return  # 함수 종료
            
            if 'entries' in result:  # 비디오 항목이 있는 경우
                videos = result['entries']  # 비디오 목록 가져오기
                playlist_name = result.get('title', 'Unknown Playlist').replace('/', '_')  # 재생목록 이름 추출
                logging.info(f"Found {len(videos)} videos in playlist: {playlist_name}")  # 비디오 수 로그 기록
                
                # CSV 파일에 제목과 링크 저장
                with open(f'{playlist_name}_videos.csv', 'w', encoding='utf-8') as csv_file:
                    csv_file.write('Title,Link\n')  # 헤더 작성
                    for video in videos:  # 비디오 목록을 순회
                        title = video.get('title', 'Unknown Title')
                        link = f"https://www.youtube.com/watch?v={video['id']}"
                        csv_file.write(f'{title},{link}\n')  # 제목과 링크 기록
                
                # 재생목록 전용 디렉토리 생성 (존재하지 않을 경우)
                playlist_dir = os.path.join('downloads', playlist_name)  # 재생목록 디렉토리 경로
                if not os.path.exists(playlist_dir):  # 디렉토리가 존재하지 않으면
                    os.makedirs(playlist_dir)  # 디렉토리 생성
                
                # 비디오 정보를 담을 큐 생성
                video_queue = Queue()  # 큐 객체 생성
                
                # 작업 스레드 생성 및 시작
                threads = []  # 스레드 목록
                for _ in range(NUM_THREADS):  # 지정된 수만큼 스레드 생성
                    t = threading.Thread(target=download_video, args=(video_queue, playlist_name, stop_event))  # 스레드 생성
                    t.daemon = True  # 데몬 스레드로 설정
                    t.start()  # 스레드 시작
                    threads.append(t)  # 스레드 목록에 추가
                
                # 비디오 정보를 큐에 추가
                for video in videos:  # 비디오 목록을 순회
                    if video:  # 비디오 정보가 있는 경우
                        video_queue.put(video)  # 큐에 비디오 정보 추가
                
                # 스레드 종료 신호를 큐에 추가
                for _ in range(NUM_THREADS):  # 각 스레드에 대해 종료 신호 추가
                    video_queue.put(None)  # None을 큐에 추가하여 스레드 종료 신호 전달
                
                # 모든 작업이 완료될 때까지 대기
                video_queue.join()  # 큐의 모든 작업이 완료될 때까지 대기
                
                # 모든 스레드가 종료될 때까지 대기
                for t in threads:  # 스레드 목록을 순회
                    t.join()  # 각 스레드가 종료될 때까지 대기
                
                logging.info(f"All downloads completed for playlist: {playlist_name}")  # 모든 다운로드 완료 로그 기록
            else:
                logging.warning("No videos found in the playlist.")  # 비디오가 없는 경우 경고 로그 기록
        except Exception as e:  # 예외 발생 시
            logging.error(f"An error occurred while crawling the playlist: {str(e)}")  # 오류 로그 기록

def get_playlist_videos(playlist_url):
    """
    재생목록의 모든 동영상 정보를 가져와서 CSV 파일에 저장하는 함수
    playlist_url: 정보를 가져올 재생목록의 URL
    """
    # yt-dlp 옵션 설정
    ydl_opts = {
        'extract_flat': 'in_playlist',  # 재생목록에서 비디오 정보만 추출
        'ignoreerrors': True,           # 오류 무시
        'quiet': True,                  # 조용한 모드
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            # 재생목록 정보 추출
            result = ydl.extract_info(playlist_url, download=False)
            
            if 'entries' in result:
                videos = result['entries']
                playlist_name = result.get('title', 'Unknown Playlist').replace('/', '_')
                logging.info(f"Found {len(videos)} videos in playlist: {playlist_name}")
                
                # CSV 파일에 제목과 링크 저장
                with open(f'{playlist_name}_videos.csv', 'w', encoding='utf-8') as csv_file:
                    csv_file.write('Title,Link\n')
                    for video in videos:
                        if video:  # 비디오 정보가 있는 경우만 처리
                            title = video.get('title', 'Unknown Title').replace(',', ' ')  # 쉼표 제거
                            video_id = video.get('id', '')
                            link = f"https://www.youtube.com/watch?v={video_id}"
                            csv_file.write(f'{title},{link}\n')
                
                logging.info(f"Successfully saved video information to {playlist_name}_videos.csv")
            else:
                logging.warning("No videos found in the playlist.")
                
        except Exception as e:
            logging.error(f"An error occurred while fetching playlist information: {str(e)}")

def main():
    """
    프로그램의 메인 함수
    """
    playlists = [
        "https://www.youtube.com/watch?v=dBNBi14AM2M&list=PLH5pXThJ-Y_A6zI3f8jr6Dg8uEQZVDWx_",  # 예시 재생목록 URL
    ]

    logging.info("Starting to process playlists")
    
    try:
        stop_event = threading.Event()
        for playlist_url in playlists:
            crawl_and_download_playlist_videos(playlist_url, stop_event)
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user")
        stop_event.set()
    
    logging.info("Finished processing playlists")

if __name__ == "__main__":
    main()