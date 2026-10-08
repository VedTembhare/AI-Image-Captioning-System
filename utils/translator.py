from deep_translator import GoogleTranslator

SUPPORTEDLANGUAGES = {
    "English": "en", "Hindi": "hi", "Marathi": "mr", "Gujarati": "gu",
    "Bengali": "bn", "Tamil": "ta", "Telugu": "te", "Kannada": "kn",
    "Malayalam": "ml", "Punjabi": "pa", "Urdu": "ur",

    "French": "fr", "Spanish": "es", "German": "de", "Italian": "it",
    "Portuguese": "pt", "Dutch": "nl", "Greek": "el", "Russian": "ru",

    "Chinese (Simplified)": "zh-cn", "Japanese": "ja", "Korean": "ko",
    "Thai": "th", "Vietnamese": "vi", "Indonesian": "id",

    "Arabic": "ar", "Turkish": "tr", "Persian": "fa",

    "Swedish": "sv", "Norwegian": "no", "Danish": "da", "Finnish": "fi"
}

def translatecaption(text, targetlanguage):
    if targetlanguage == "English":
        return text
    try:
        return GoogleTranslator(source="auto", target=SUPPORTEDLANGUAGES[targetlanguage]).translate(text)
    except:
        return text