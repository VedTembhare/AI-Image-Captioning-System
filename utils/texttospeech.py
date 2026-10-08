from gtts import gTTS
import tempfile

def generateaudio(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp.name)
    return temp.name