# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import threading
import pyttsx3

# Initialize Flask app
app = Flask(__name__)

# Read appointment data from CSV file
appointments_df = pd.read_csv("appointment_data.csv")

# Feature engineering: Extract features from appointment data
appointments_df['weekday'] = pd.to_datetime(appointments_df['date'], format='%m-%d-%y').dt.dayofweek
appointments_df['hour'] = pd.to_datetime(appointments_df['time'], format='%H:%M').dt.hour

# Define features and target variable
X = appointments_df[['weekday', 'hour', 'duration']]
y = appointments_df['status']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)

# Appointment scheduling functions remain the same as before

def check_availability_ml(date, time, duration, participants):
    # Extract features from input data
    weekday = pd.to_datetime(date, format='%m-%d-%y').dayofweek
    hour = pd.to_datetime(time, format='%H:%M').hour

    # Make prediction using the trained model
    prediction = rf_classifier.predict([[weekday, hour, duration]])

    if prediction[0] == 'confirmed':
        return True, "Appointment scheduled successfully!"
    else:
        return False, "I regret to inform you that the meeting you had scheduled has been canceled due to unexpected circumstances. We are currently working on finding a suitable time and will notify you as soon as possible about the rescheduled meeting. In the meantime, if you need any assistance or have any questions, please feel free to let me know. I apologize for any inconvenience this may have caused. Thank you for your understanding."

def send_reminder_notification(user, appointment_details):
    # Logic to send reminder notification
    print(f"Reminder: You have an appointment {appointment_details}.")

def notify_reminder(participant, appointment_details):
    # This function will be run in a separate thread to send reminder notifications
    threading.Timer(10, send_reminder_notification, args=[participant, appointment_details]).start()

def speak_message(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

# Define routes
@app.route('/')
def index():
    return render_template('appointment.html')

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
            for participant in participants:
                notify_reminder(participant, appointment_details)
            return render_template('success.html', message="Appointment scheduled successfully!")
        else:
            speak_message(message)
            return render_template('failure.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
