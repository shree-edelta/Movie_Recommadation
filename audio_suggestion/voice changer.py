import boto3

# Initialize the Polly client
polly = boto3.client('polly')

# Get a voice (e.g., a child voice)
response = polly.synthesize_speech(
    Text='नमस्ते! आप कैसे हैं?',
    VoiceId='Aditi',  # Example: 'Aditi' is the Hindi female voice. Polly also has a 'Raveena' voice for a more child-like voice.
    OutputFormat='mp3'
)

# Save to file
with open("output.mp3", "wb") as file:
    file.write(response['AudioStream'].read())
