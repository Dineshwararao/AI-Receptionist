from flask import Flask, render_template, Response
import cv2
import pyttsx3
import threading

from flask import Flask, render_template, request, redirect, url_for , jsonify
from datetime import datetime
import cv2
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import threading
import pyttsx3
import speech_recognition as sr
from nltk.chat.util import Chat, reflections
import random
import qrcode
from datetime import datetime
import pandas as pd

from flask import Flask

app = Flask(__name__, static_url_path='/static', static_folder='static')


class Visitor:
    def __init__(self, name, company, purpose, check_in_time, image_path):
        self.name = name
        self.company = company
        self.purpose = purpose
        self.check_in_time = check_in_time
        self.image_path = image_path

# Modify the Receptionist class to include greeting functionality
class Receptionist:
    def __init__(self):
        self.visitors = []

    def check_in_visitor(self, name, company, purpose):
        check_in_time = datetime.now()

        # Capture visitor's face using OpenCV
        camera = cv2.VideoCapture(0)
        return_value, image = camera.read()
        
        # Specify the desired path for saving the image file
        image_folder = "static/images"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_path = os.path.join(image_folder, f"{name}_{check_in_time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
        cv2.imwrite(image_path, image)
        del(camera)

        visitor = Visitor(name, company, purpose, check_in_time, image_path)
        self.visitors.append(visitor)
        print(f"{visitor.name} from {visitor.company} has checked in at {visitor.check_in_time}")

        # Save visitor information to a text file
        with open('visitor_info.txt', 'a') as file:
            file.write(f"Name: {visitor.name}, Company: {visitor.company}, Purpose: {visitor.purpose}, Check-in Time: {visitor.check_in_time}, Image Path: {visitor.image_path}\n")

       

# Read appointment data from CSV file with date format 'DD-MM-YYYY'
appointments_df = pd.read_csv("appointment_data.csv", parse_dates=['date'], dayfirst=True)

# Feature engineering: Extract features from appointment data
appointments_df['weekday'] = appointments_df['date'].dt.dayofweek
appointments_df['hour'] = pd.to_datetime(appointments_df['time'], format='%H:%M').dt.hour

# Function to check availability using machine learning model
def check_availability_ml(date, time, duration, participants):
    # Convert date string to datetime object with 'YYYY-MM-DD' format
    date = datetime.strptime(date, '%Y-%m-%d')
    
    # Adjust the date format to 'YYYY-MM-DD' for parsing
    weekday = date.weekday()
    hour = pd.to_datetime(time, format='%H:%M').hour

    # Make prediction using the trained model
    prediction = rf_classifier.predict([[weekday, hour, duration]])

    if prediction[0] == 'confirmed':
        return True, "Appointment scheduled successfully!"
    else:
        message = "I regret to inform you that the meeting you had scheduled has been canceled due to unexpected circumstances. We are currently working on finding a suitable time and will notify you as soon as possible about the rescheduled meeting. In the meantime, if you need any assistance or have any questions, please feel free to let me know. I apologize for any inconvenience this may have caused. Thank you for your understanding."
        speak_message(message)  # Speaking the cancellation message
        return False, message


# Define features and target variable
X = appointments_df[['weekday', 'hour', 'duration']]
y = appointments_df['status']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)

# Placeholder for sending reminder notification
def send_reminder_notification(user, appointment_details):
    # Logic to send reminder notification
    print(f"Reminder: You have an appointment {appointment_details}.")

# Placeholder for speaking messages
def speak_message(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

def generate_qr_code(appointment_details):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(appointment_details)
    qr.make(fit=True)

    qr_code_img = qr.make_image(fill_color="black", back_color="white")
    qr_code_img_path = os.path.join("static", "appointment_qr_code.png")
    qr_code_img.save(qr_code_img_path)
    print("QR code generated successfully.")
    return qr_code_img_path


# Define pairs for each department
department_pairs = {
    "sales": [
        [
            r"where is the (sales department|section|division)",
            ["The sales department is located on the 5th floor. Would you like directions?"]
        ],
        [
            r"(directions|map) for (sales department|section|division)",
            ["The sales department is located on the 5th floor. You can take the elevator to the 5th floor and follow the signs."]
        ]
    ],
    "marketing": [
        [
            r"where is the (marketing department|section|division)",
            ["The marketing department is located on the 3rd floor. Would you like directions?"]
        ],
        [
            r"(directions|map) for (marketing department|section|division)",
            ["The marketing department is located on the 3rd floor. You can take the stairs or elevator to the 3rd floor and proceed to the marketing department."]
        ]
    ],
    "finance": [
        [
            r"where is the (finance department|section|division)",
            ["The finance department is located on the 2nd floor. Would you like directions?"]
        ],
        [
            r"(directions|map) for (finance department|section|division)",
            ["The finance department is located on the 2nd floor. You can take the stairs or elevator to the 2nd floor and proceed to the finance department."]
        ]
    ],
    "hr": [
        [
            r"where is the (hr|human resources department|section|division)",
            ["The HR department is located on the 4th floor. Would you like directions?"]
        ],
        [
            r"(directions|map) for (hr|human resources department|section|division)",
            ["The HR department is located on the 4th floor. You can take the elevator to the 4th floor and follow the signs to the HR department."]
        ]
    ],
    "accounting": [
        [
            r"where is the (accounting department|section|division)",
            ["The accounting department is located on the 6th floor. Would you like directions?"]
        ],
        [
            r"(directions|map) for (accounting department|section|division)",
            ["The accounting department is located on the 6th floor. You can take the elevator to the 6th floor and proceed to the accounting department."]
        ]
    ]
}

# Merge all department pairs into a single list
all_department_pairs = [pair for department_pair in department_pairs.values() for pair in department_pair]

# Merge department pairs with other pairs
pairs = all_department_pairs + [
    # General pairs for other inquiries
    [
        r"hi|hello|hey",
        ["Hello!", "Hey there!", "Hi! How can I assist you today?"]
    ],
    [
        r"what are you|who are you",
        ["I am an AI receptionist, here to help.", "I am a new AI model designed to assist the company."]
    ],
    [
        r"how are you|how are you doing",
        ["I'm doing well, thank you!", "I'm here to assist you, so I'm doing great!"]
    ],
    [
        r"what can you do|what do you do",
        ["I can help you with various tasks such as answering questions, providing information, and more."]
    ],
    [
        r"what's your name",
        ["You can call me AI receptionist.", "I'm AI receptionist, your virtual assistant."]
    ],
    [
        r"bye|goodbye",
        ["Goodbye!", "See you later!", "Take care!"]
    ],
    [
        r"thank you|thanks",
        ["You're welcome!", "My pleasure!", "Anytime!"]
    ],
    [
        r"(.*) your name(.*)",
        ["You can call me AI receptionist.", "I'm AI receptionist, your virtual assistant."]
    ],
    [
        r"(.*) (weather|temperature) (.*)",
        ["I'm sorry, I can't provide weather information at the moment."]
    ],
    [ 
        r""

    ]
]

# Function to respond to user queries
def ai_response(user_input):
    chatbot = Chat(pairs, reflections)
    response = chatbot.respond(user_input)
    return response

# Function to convert speech to text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak:")
        audio = recognizer.listen(source)
    try:
        user_input = recognizer.recognize_google(audio)
        return user_input
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return "Could not request results; {0}".format(e)

# Function to generate AI response using text or speech input
def generate_response(input_text):
    if input_text.startswith('!speech'):
        return speech_to_text()
    else:
        return ai_response(input_text)


camera = cv2.VideoCapture(0)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Say "Hello" when a face is detected
        threading.Thread(target=say_hello).start()
    return frame

def say_hello():
    # Say "Hello" using text-to-speech engine
    engine.say("Hello!")
    engine.runAndWait()

def generate_frames():
    while True:
        # Read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)
            # Detect faces
            frame_with_faces = detect_faces(frame)
            ret, buffer = cv2.imencode('.jpg', frame_with_faces)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Route for rendering the index page
@app.route("/indexmain.html")
def indexmain():
    return render_template("indexmain.html")

# Route for rendering the general conversation page
@app.route("/generalcon.html")
def general_conversation():
    return render_template("generalcon.html")

# Route for rendering the feedback form
@app.route("/feedback.html", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        feedback_data = {
            "overall_rating": request.form["overall_rating"],
            "scheduling_rating": request.form["scheduling_rating"],
            "friendliness_rating": request.form["friendliness_rating"],
            "comments": request.form["comments"]
        }
        collect_feedback(feedback_data)
        return render_template("thankyou.html")
    else:
        return render_template("feedback.html")

# Function to collect feedback and store it in a file
def collect_feedback(feedback):
    with open("feedback.txt", "a") as file:
        file.write("Overall Experience Rating: {} stars\n".format(feedback["overall_rating"]))
        file.write("Appointment Scheduling Rating: {} stars\n".format(feedback["scheduling_rating"]))
        file.write("Friendliness Rating: {} stars\n".format(feedback["friendliness_rating"]))
        file.write("Comments: {}\n".format(feedback["comments"]))
        file.write("\n")

# Route for rendering the thank you page
@app.route("/thankyou.html")
def thankyou():
    return render_template("thankyou.html")

# Route for handling user input and generating responses
@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form["user_input"]
    response = generate_response(user_input)
    return {"response": response}

# Route for rendering the appointment page
@app.route("/appointment.html")
def appointment():
    return render_template("appointment.html")

# Route for handling appointment scheduling
@app.route('/schedule_appointment', methods=['POST'])
def schedule_appointment():
    if request.method == 'POST':
        date = request.form['date']
        time = request.form['time']
        duration = int(request.form['duration'])
        participants = request.form['participants'].split(',')
        room = request.form['room']

        is_available, message = check_availability_ml(date, time, duration, participants)
        if is_available:
            appointment_details = f"on {date} at {time} for {duration} minutes in {room}"
            qr_code_img_path = generate_qr_code(appointment_details)  # Generate QR code and get image path
            for participant in participants:
                send_reminder_notification(participant, appointment_details)
            return render_template('success.html', message="Appointment scheduled successfully!", appointment_details=appointment_details, qr_code_img_path=qr_code_img_path)
        else:
            return render_template('failure.html', message=message)


# Route for rendering the visitor check-in page
@app.route('/visitor.html')
def visitor():
    return render_template('visitor.html')

# Route for handling visitor check-in
@app.route('/check_in', methods=['POST'])
def check_in():
    name = request.form['name']
    company = request.form['company']
    purpose = request.form['purpose']

    receptionist = Receptionist()
    receptionist.check_in_visitor(name, company, purpose)

    return redirect(url_for('visitor'))


if __name__ == "__main__":
    app.run(debug=True)