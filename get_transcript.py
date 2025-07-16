from youtube_transcript_api import YouTubeTranscriptApi

yt_url = "https://www.youtube.com/watch?v=0dZMAVOA8Xo"
video_id = yt_url.split("=")[1]

ytt_api = YouTubeTranscriptApi()

# Get the list of available transcripts
transcript_list = ytt_api.list_transcripts(video_id)

# Try to get the English transcript
try:
    en_transcript = transcript_list.find_transcript(['en'])
    transcript = en_transcript.fetch()
    print("English transcript found:"   + transcript_list)
except Exception:
    # If English transcript is not available, try to translate to English
    print("English transcript not found, trying to translate...")
    try:
        # Get the first available transcript and translate to English
        first_transcript = list(transcript_list)[0]
        transcript = first_transcript.translate('en').fetch()
        print("Translated transcript to English:")
    except Exception as e:
        print("Could not fetch or translate transcript:", e)
        transcript = []

print(transcript)