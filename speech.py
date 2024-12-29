import pyttsx3

# Function to convert text file to speech
def text_to_speech(file_path):
    try:
        # Read the contents of the text file
        with open(file_path, 'r') as file:
            text = file.read()

        if not text.strip():
            print("The file is empty. Nothing to convert to speech.")
            return

        # Initialize the text-to-speech engine
        engine = pyttsx3.init()

        # Set properties for the voice (optional)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)  # Choose voice (0: male, 1: female, etc.)
        engine.setProperty('rate', 125)  # Speed of speech
        engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

        # Speak the text
        print("Converting text to speech...")
        engine.say(text)
        engine.runAndWait()
        print("Text-to-speech conversion completed.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    file_path = "detected_letters.txt"  # File containing the detected letters
    text_to_speech(file_path)
