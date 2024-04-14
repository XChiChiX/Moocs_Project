import whisper
import os

def CountParagraphs():
    # Define the directory path
    directory_path = './data/ans'

    # Define the keyword
    keyword = 'Summary'

    # Initialize a counter for the files that contain the keyword
    count = 0

    # List all files in the directory
    files = os.listdir(directory_path)

    # Iterate over the files
    for file in files:
        # Check if the keyword is in the filename
        if keyword in file:
            # Increment the counter
            count += 1

    # Print the number of files that contain the keyword
    return count

def start():
    model = whisper.load_model("base")

    for i in range(1 , CountParagraphs() + 1):
        audio = whisper.load_audio("./data/ans/Summary"+str(i)+".mp3")
        result = model.transcribe(audio)

        Summary_path = "./data/ans/Summary"+str(i)+".txt"

        f = open(Summary_path,"w", encoding='UTF-8')

        for segment in result["segments"]:
            f.write(segment['text'].replace(' ', '')+" "+(str)(round(segment['start'], 2))+" "+(str)(round(segment['end'], 2)))#float/float/str
            f.write("\n")

        f.close()
        print("Summary"+str(i)+".txt finish !")

        audio = whisper.load_audio("./data/ans/Question"+str(i)+".mp3")
        result = model.transcribe(audio)

        Question_path = "./data/ans/Question"+str(i)+".txt"

        f = open(Question_path,"w", encoding='UTF-8')

        for segment in result["segments"]:
            f.write(segment['text'].replace(' ', '')+" "+(str)(round(segment['start'], 2))+" "+(str)(round(segment['end'], 2)))#float/float/str
            f.write("\n")

        f.close()
        print("Question"+str(i)+".txt finish !")

