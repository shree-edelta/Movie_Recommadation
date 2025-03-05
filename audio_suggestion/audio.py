from gtts import gTTS
import os

text = "एक समय की बात है, एक छोटे से प्यारे बिल्ली के बच्चे का नाम था मिलो। आज उसका जन्मदिन था! घर में गुब्बारे, उपहार और एक बड़ी सी स्वादिष्ट केक रखी थी। उसकी माँ प्यार से मुस्कुरा रही थी, और उसके दोस्त तालियां बजा रहे थे।"
tts = gTTS(text=text, lang='hi', slow=False)
tts.save("output.mp3")

os.system("start output.mp3")