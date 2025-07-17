from youtube_transcript_api import YouTubeTranscriptApi

# function for fetching transcript

def fetch_transcript(yt_url):
    # Extract the video ID from the URL
    video_id = yt_url.split("=")[1]

    ytt_api = YouTubeTranscriptApi()

    try:
        # Get the list of available transcripts
        transcript_list = ytt_api.list_transcripts(video_id)
        # print(transcript_list)

        # get the transcript in english
        transcript = transcript_list.find_transcript(['bn','en'])

        # get the transcript in text format
        raw_transcript = transcript.fetch().to_raw_data()
        # print(raw_transcript)
        full_transcript_with_time = ''
        # go through the list and print the text with time
        for i in raw_transcript:
            transcript_with_time = f'{i["start"]}s: {i["text"]}'
            # print(transcript_with_time)
            full_transcript_with_time += transcript_with_time + '\n'
        # print(full_transcript_with_time)
        return full_transcript_with_time

        
    except Exception as e:
        print(f"Error: {e}")
    
# yt_url = "https://www.youtube.com/watch?v=0nhkU_DImhU"
# transcript = fetch_transcript(yt_url)
