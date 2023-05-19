import speech_recognition as sr

def Listen():  
    r = sr.Recognizer()

    with sr.Microphone(device_index=0) as source:
    
        print("Listening....")
        r.pause_threshold=1
        audio = r.listen(source,0,5)
        
    try:
        print("Recognizing...")
        query = r.recognize_google(audio,language="en-in")
        print(f"You Said: {query}")
    
    except:
        return ""

    query = str(query)
    return query.lower()

 

