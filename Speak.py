import pyttsx3 #used to convert text to speech

def Say(Text):
    engine = pyttsx3.init("sapi5") #microsoft speaking API
#We have make a variable and store it here
    voices = engine.getProperty('voices')  #we are assccessing voices inside engine
    engine.setProperty('voices',voices[0].id)
    engine.setProperty('rate',170)#85% set
    print("    ")
    print(f"A.I : {Text}")
    engine.say(text=Text)
    engine.runAndWait()
    print("    ")

#by the help of this function our voice assistant BOSS will speak
