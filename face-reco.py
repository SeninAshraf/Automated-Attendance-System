from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
import sys
import sqlite3
from datetime import date, datetime
import cv2, os, numpy
import requests
import time
from ultralytics import YOLO

ui, _ = loadUiType('face-reco.ui')

class MainApp(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.tabWidget.setCurrentIndex(0)
        self.LOGINBUTTON.clicked.connect(self.login)
        self.LOGOUTBUTTON.clicked.connect(self.logout)
        self.CLOSEBUTTON.clicked.connect(self.close_window)
        self.TRAINLINK1.clicked.connect(self.show_training_form)
        self.ATTLINK1.clicked.connect(self.show_attendance_form)
        self.REPORTSLINK1.clicked.connect(self.show_report_form)
        self.TRAININGBACK.clicked.connect(self.show_mainform)
        self.ATTENDANCEBACK.clicked.connect(self.show_mainform)
        self.REPORTSBACK.clicked.connect(self.show_mainform)
        self.TRAININGBUTTON.clicked.connect(self.start_training)
        self.RECORD_2.clicked.connect(self.record_attendance)
        self.dateEdit.setDate(date.today())
        self.dateEdit.dateChanged.connect(self.selected_date)
        self.tabWidget.setStyleSheet("QTabWidget::pane{border:0;}")
        self.BOT_TOKEN = "7340973121:AAHlzfz3vhwH1jGDyt5t1ms8tbXAwiXHV4k"
        self.CHAT_ID = "1255081338"
        
        # Initialize YOLO model for ID strap detection
        try:
            # Update the path to your trained YOLO model
            self.id_model = YOLO("best.pt")
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.id_model = None

        # Ensure the required directories exist
        self.required_dirs = ["reports", "captured_images"]
        for directory in self.required_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

        # Ensure consistent database name
        self.DB_NAME = "face-reco.db"

        try:
            con = sqlite3.connect(self.DB_NAME)
            con.execute("CREATE TABLE IF NOT EXISTS attendance(attendanceid INTEGER,name TEXT,attendancedate TEXT)")
            con.commit()
            print("Table created successfully")
            con.close()
        except Exception as e:
            print(f"Error in database initialization: {e}")

    ### LOGIN PROCESS ###
    def login(self):
        pw = self.PASSWORD.text()
        if pw == "123":
            self.PASSWORD.setText("")
            self.tabWidget.setCurrentIndex(1)
        else:
            self.LOGININFO.setText("Invalid Password...")
            self.PASSWORD.setText("")

    ### LOG OUT PROCESS ###
    def logout(self):
        self.tabWidget.setCurrentIndex(0)

    ### CLOSE WINDOW PROCESS ###
    def close_window(self):
        self.close()

    ### SHOW MAIN FORM ###
    def show_mainform(self):
        self.tabWidget.setCurrentIndex(1)

    ### SHOW TRAINING FORM ###
    def show_training_form(self):
        self.tabWidget.setCurrentIndex(2)

    ### SHOW ATTENDANCE FORM ###
    def show_attendance_form(self):
        self.tabWidget.setCurrentIndex(3)

    ### SHOW REPORT FORM ###
    def show_report_form(self):
        self.tabWidget.setCurrentIndex(4)
        self.REPORTS.setRowCount(0)
        self.REPORTS.clear()
        
        try:
            con = sqlite3.connect(self.DB_NAME)
            cursor = con.execute("SELECT * FROM attendance")
            result = cursor.fetchall()
            
            r = 0
            c = 0
            for row_number, row_data in enumerate(result):
                r += 1
                c = 0
                for column_number, data in enumerate(row_data):
                    c += 1
                    
            self.REPORTS.setColumnCount(c)
            for row_number, row_data in enumerate(result):
                self.REPORTS.insertRow(row_number)
                for column_number, data in enumerate(row_data):
                    self.REPORTS.setItem(row_number, column_number, QTableWidgetItem(str(data)))
                    
            self.REPORTS.setHorizontalHeaderLabels(['Id', 'Name', 'Date'])
            self.REPORTS.setColumnWidth(0, 50)
            self.REPORTS.setColumnWidth(1, 60)
            self.REPORTS.setColumnWidth(2, 100)
            self.REPORTS.verticalHeader().setVisible(False)
            
            con.close()
        except Exception as e:
            print(f"Error loading reports: {e}")
            QMessageBox.warning(self, "Database Error", f"Error loading reports: {e}")

    ### SHOW SELECTED DATE REPORT ###
    def selected_date(self):
        self.REPORTS.setRowCount(0)
        self.REPORTS.clear()
        
        try:
            con = sqlite3.connect(self.DB_NAME)
            cursor = con.execute("SELECT * FROM attendance WHERE attendancedate = '"+ str((self.dateEdit.date()).toPyDate())+"'")
            result = cursor.fetchall()
            
            r = 0
            c = 0
            for row_number, row_data in enumerate(result):
                r += 1
                c = 0
                for column_number, data in enumerate(row_data):
                    c += 1
                    
            self.REPORTS.setColumnCount(c)
            for row_number, row_data in enumerate(result):
                self.REPORTS.insertRow(row_number)
                for column_number, data in enumerate(row_data):
                    self.REPORTS.setItem(row_number, column_number, QTableWidgetItem(str(data)))
                    
            self.REPORTS.setHorizontalHeaderLabels(['Id', 'Name', 'Date'])
            self.REPORTS.setColumnWidth(0, 50)
            self.REPORTS.setColumnWidth(1, 60)
            self.REPORTS.setColumnWidth(2, 100)
            self.REPORTS.verticalHeader().setVisible(False)
            
            con.close()
        except Exception as e:
            print(f"Error filtering reports by date: {e}")
            QMessageBox.warning(self, "Database Error", f"Error filtering reports: {e}")

    ### TRAINING PROCESS ###
    def start_training(self):
        try:
            haar_file = 'haarcascade_frontalface_default.xml'
            
            # Check if Haar cascade file exists
            if not os.path.exists(haar_file):
                QMessageBox.warning(self, "Error", f"Haar cascade file not found at: {haar_file}")
                return
        
            # Ensure Haar cascade is loaded correctly
            face_cascade = cv2.CascadeClassifier(haar_file)
            if face_cascade.empty():
                QMessageBox.warning(self, "Error", "Failed to load Haar cascade XML file.")
                return
            
            # Get name and image count
            datasets = 'datasets'
            sub_data = self.TraineName.text().strip()  # Strip to remove extra spaces
        
            if not sub_data:
                QMessageBox.warning(self, "Error", "Please enter a name for training.")
                return
        
            # Create datasets directory if it doesn't exist
            if not os.path.exists(datasets):
                os.makedirs(datasets)
    
            path = os.path.join(datasets, sub_data)
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"New directory created: {path}")
        
            (width, height) = (130, 100)  # Resize dimensions for face images
            webcam = cv2.VideoCapture(0)
    
            if not webcam.isOpened():
                QMessageBox.warning(self, "Error", "Webcam not detected. Please check your camera connection.")
                return
            
            # Get image count
            max_count_text = self.TraineCount.text()
            if not max_count_text.isdigit():
                QMessageBox.warning(self, "Error", "Please enter a valid number of images.")
                return
            
            max_count = int(max_count_text)
            count = 1
    
            # Notify training start
            self.send_telegram_message(f"Starting face training for: {sub_data} with {max_count} images")
    
            while count <= max_count:
                ret, im = webcam.read()
                if not ret:
                    QMessageBox.warning(self, "Error", "Failed to capture image from webcam.")
                    break
    
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    
                for (x, y, w, h) in faces:
                    cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face = gray[y:y + h, x:x + w]
                    face_resize = cv2.resize(face, (width, height))
                    cv2.imwrite(f'{path}/{count}.png', face_resize)
                    count += 1
    
                cv2.imshow('Training Face Capture', im)
                if cv2.waitKey(10) == 27:  # Press 'ESC' to exit
                    break
    
            webcam.release()
            cv2.destroyAllWindows()
    
            QMessageBox.information(self, "Attendance System", "Training Completed Successfully")
            self.TraineName.setText("")
            self.TraineCount.setText("100")
            
            # Notify training completion
            self.send_telegram_message(f"Face training completed for: {sub_data}")
            
        except Exception as e:
            print(f"Error in training: {e}")
            QMessageBox.warning(self, "Training Error", f"An error occurred during training: {e}")
            self.send_telegram_message(f"Error during face training for {sub_data}: {str(e)}")

    ### RECORD ATTENDANCE - INTEGRATED WITH ID STRAP DETECTION ###
    def record_attendance(self):
        self.currentprocess.setText("Process started.. Waiting for face recognition...")
        
        try:
            # Step 1: Setup and validation
            haar_file = 'haarcascade_frontalface_default.xml'
            
            # Check if Haar cascade file exists
            if not os.path.exists(haar_file):
                error_msg = f"Haar cascade file not found at: {haar_file}"
                self.currentprocess.setText(error_msg)
                QMessageBox.warning(self, "Error", error_msg)
                return
                
            face_cascade = cv2.CascadeClassifier(haar_file)
            if face_cascade.empty():
                error_msg = "Failed to load Haar cascade XML file."
                self.currentprocess.setText(error_msg)
                QMessageBox.warning(self, "Error", error_msg)
                return
                
            # Check if YOLO model is loaded (we'll use it later)
            if self.id_model is None:
                error_msg = "ID strap detection model not loaded. Cannot proceed."
                self.currentprocess.setText(error_msg)
                QMessageBox.warning(self, "Error", error_msg)
                self.send_telegram_message("ERROR: ID strap detection model not loaded")
                return
                
            # Check for training data
            datasets = 'datasets'
            if not os.path.exists(datasets) or len(os.listdir(datasets)) == 0:
                error_msg = "No training data available. Please train faces first."
                self.currentprocess.setText(error_msg)
                QMessageBox.warning(self, "Error", error_msg)
                return
                
            # Read training data
            (images, labels, names, id) = ([], [], {}, 0)
            for (subdirs, dirs, files) in os.walk(datasets):
                for subdir in dirs:
                    names[id] = subdir
                    subjectpath = os.path.join(datasets, subdir)
                    for filename in os.listdir(subjectpath):
                        path = subjectpath + "/" + filename
                        label = id
                        images.append(cv2.imread(path, 0))
                        labels.append(int(label))
                    id += 1
                    
            if len(images) == 0:
                error_msg = "No training images found. Please train faces first."
                self.currentprocess.setText(error_msg)
                QMessageBox.warning(self, "Error", error_msg)
                return
                
            # Train face recognition model
            (images, labels) = [numpy.array(lis) for lis in [images, labels]]
            print(f"Training with {len(images)} images")
            
            (width, height) = (130, 100)
            model = cv2.face.LBPHFaceRecognizer_create()
            model.train(images, labels)
    
            # Initialize webcam
            webcam = cv2.VideoCapture(0)
            if not webcam.isOpened():
                error_msg = "Error: Webcam not found."
                self.currentprocess.setText(error_msg)
                QMessageBox.warning(self, "Error", error_msg)
                return
    
            # Variables for tracking the process
            face_detected = False
            face_recognized = False
            person_name = "Unknown"
            id_strap_detected = False
            
            # Notify attendance process start
            self.send_telegram_message("Attendance recording process started")
            
            # STAGE 1: Face Recognition Only
            self.currentprocess.setText("STAGE 1: Waiting for face recognition...")
            max_face_frames = 100  # Limit frames to prevent indefinite loop
            face_frame_count = 0
            face_recognition_complete = False
            
            while face_frame_count < max_face_frames and not face_recognition_complete:
                face_frame_count += 1
                ret, im = webcam.read()
                if not ret:
                    error_msg = "Error: Failed to capture image from webcam."
                    self.currentprocess.setText(error_msg)
                    QMessageBox.warning(self, "Error", error_msg)
                    break
                
                # Face Recognition
                display_frame = im.copy()
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    face_detected = True
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                        face = gray[y:y + h, x:x + w]
                        face_resize = cv2.resize(face, (width, height))
                        prediction = model.predict(face_resize)
                        
                        # Check the confidence score
                        if prediction[1] < 800:  # Adjust threshold as needed
                            # Recognized face
                            face_recognized = True
                            person_name = names[prediction[0]]
                            cv2.putText(display_frame, f'{person_name}-{prediction[1]:.0f}', 
                                    (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
                            self.currentprocess.setText(f"Face recognized as {person_name}! Moving to ID card detection...")
                            face_recognition_complete = True
                        else:
                            # Unknown face
                            cv2.putText(display_frame, 'Unknown', (x - 10, y - 10), 
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                            self.currentprocess.setText("Unknown face detected! Please try again.")
                
                # Status display on frame
                status_text = "STAGE 1: Face Recognition"
                cv2.putText(display_frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if face_recognized:
                    status_text = f"Face Recognized: {person_name} ✅"
                else:
                    status_text = "Face: Not recognized ❌"
                
                cv2.putText(display_frame, status_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show current frame with detections
                cv2.imshow("Attendance System", display_frame)
                
                # Process key input
                key = cv2.waitKey(10)
                if key == 27:  # ESC key to exit
                    break
                
                # If face recognized, save image and break out of this stage
                if face_recognized:
                    # Save the recognized face image
                    timestamp = int(time.time())
                    face_image_filename = f"captured_images/face_{person_name}_{timestamp}.jpg"
                    cv2.imwrite(face_image_filename, display_frame)
                    self.send_telegram_photo(
                        face_image_filename, 
                        f"✅ STAGE 1 Complete: Face recognized as {person_name}. Now checking for ID card..."
                    )
                    break
            
            # If no face recognized, exit
            if not face_recognized:
                self.currentprocess.setText("No face recognized. Attendance process aborted.")
                webcam.release()
                cv2.destroyAllWindows()
                self.send_telegram_message("❌ Attendance process aborted: No face recognized")
                return
            
            # STAGE 2: ID Strap Detection (only if face was recognized)
            self.currentprocess.setText(f"STAGE 2: Face recognized as {person_name}. Now checking for ID card...")
            max_id_frames = 100  # Limit frames for ID detection
            id_frame_count = 0
            
            while id_frame_count < max_id_frames and not id_strap_detected:
                id_frame_count += 1
                ret, im = webcam.read()
                if not ret:
                    error_msg = "Error: Failed to capture image from webcam."
                    self.currentprocess.setText(error_msg)
                    QMessageBox.warning(self, "Error", error_msg)
                    break
                
                # Create display frame
                display_frame = im.copy()
                
                # Always show the recognized face info
                cv2.putText(display_frame, f"Face Recognized: {person_name} ✅", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # ID Strap Detection
                rgb_frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                results = self.id_model(rgb_frame)
                
                # Process YOLO results
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()
                    label = results[0].names[int(box.cls[0].item())]
                    
                    # Draw bounding box for ID strap
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    if confidence > 0.2:  # Adjust threshold as needed
                        id_strap_detected = True
                        self.currentprocess.setText(f"ID strap detected with confidence: {confidence:.2f}")
                
                # Status display on frame
                cv2.putText(display_frame, "STAGE 2: ID Card Detection", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if id_strap_detected:
                    status_text = "ID Card: Detected ✅"
                else:
                    status_text = "ID Card: Looking for ID card... ❌"
                
                cv2.putText(display_frame, status_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show current frame with detections
                cv2.imshow("Attendance System", display_frame)
                
                # Process key input
                key = cv2.waitKey(10)
                if key == 27:  # ESC key to exit
                    break
                
                # If ID strap detected, break out of this stage
                if id_strap_detected:
                    # Save the recognized ID card image
                    timestamp = int(time.time())
                    id_image_filename = f"captured_images/id_{person_name}_{timestamp}.jpg"
                    cv2.imwrite(id_image_filename, display_frame)
                    self.send_telegram_photo(
                        id_image_filename, 
                        f"✅ STAGE 2 Complete: ID card detected for {person_name}. Processing attendance..."
                    )
                    break
            
            # STAGE 3: Process Attendance based on results
            if face_recognized and id_strap_detected:
                self.currentprocess.setText(f"STAGE 3: Processing attendance for {person_name} with ID card...")
                
                # Check if attendance already registered today
                attendanceid = 0
                available = False
                
                try:
                    # Get next attendance ID
                    connection = sqlite3.connect(self.DB_NAME)
                    cursor = connection.execute("SELECT MAX(attendanceid) from attendance")
                    result = cursor.fetchall()
                    if result and result[0][0] is not None:
                        attendanceid = int(result[0][0]) + 1
                    else:
                        attendanceid = 1
                    connection.close()
                    
                    # Check if already registered today
                    con = sqlite3.connect(self.DB_NAME)
                    cursor = con.execute("SELECT * FROM attendance WHERE name='" + 
                                      str(person_name) + "' and attendancedate = '" + 
                                      str(date.today()) + "'")
                    result = cursor.fetchall()
                    if result:
                        available = True
                        self.currentprocess.setText(f"{person_name} already has attendance for today")
                        # Send notification about duplicate attendance attempt
                        self.send_telegram_message(
                            f"Note: {person_name} attempted to register attendance again on {date.today()}"
                        )
                    
                    # Register attendance if not already done
                    if not available:
                        con.execute("INSERT INTO attendance VALUES(" + 
                                 str(attendanceid) + ",'" + 
                                 str(person_name) + "','" + 
                                 str(date.today()) + "')")
                        con.commit()
                        
                        # Save the final image with timestamp
                        timestamp = int(time.time())
                        final_image_filename = f"captured_images/final_{person_name}_{timestamp}.jpg"
                        cv2.imwrite(final_image_filename, display_frame)
                        
                        # Save attendance report
                        report_path = f"reports/attendance_report_{timestamp}.txt"
                        with open(report_path, "w") as file:
                            file.write(f"Attendance ID: {attendanceid}\n")
                            file.write(f"Name: {person_name}\n")
                            file.write(f"Date: {date.today()}\n")
                            file.write(f"Time: {datetime.now().strftime('%H:%M:%S')}\n")
                            file.write("Attendance Status: Present\n")
                            file.write("ID Card Status: Wearing ✅\n")
                            file.write("Processing: Sequential verification successful\n")
                        
                        # Send success notification with image
                        self.send_telegram_photo(
                            final_image_filename, 
                            f"✅ ATTENDANCE RECORDED: {person_name} on {date.today()} at {datetime.now().strftime('%H:%M:%S')}\n"\
                            f"Face Recognition: ✅\n"\
                            f"ID Card Detection: ✅\n"\
                            f"Sequential Verification: ✅"
                        )
                        
                        # Also send the report file
                        self.send_telegram_file(
                            report_path,
                            f"Attendance Report for {person_name}"
                        )
                        
                        self.currentprocess.setText(f"Success! Attendance recorded for {person_name} (with ID card)")
                    
                    con.close()
                
                except sqlite3.Error as e:
                    print(f"Error in database: {e}")
                    self.currentprocess.setText(f"Database error: {str(e)}")
                    self.send_telegram_message(
                        f"ERROR: Failed to record attendance for {person_name} - Database error"
                    )
            
            elif face_recognized and not id_strap_detected:
                self.currentprocess.setText(f"Face recognized as {person_name} but NO ID card detected. Attendance NOT registered.")
                timestamp = int(time.time())
                error_image_filename = f"captured_images/no_id_{person_name}_{timestamp}.jpg"
                cv2.imwrite(error_image_filename, display_frame)
                self.send_telegram_photo(
                    error_image_filename, 
                    f"⚠️ ALERT: {person_name} detected without ID card on {date.today()} at {datetime.now().strftime('%H:%M:%S')}\n"\
                    f"Face Recognition: ✅\n"\
                    f"ID Card Detection: ❌\n"\
                    f"Attendance NOT registered!"
                )
            
            # Close resources
            webcam.release()
            cv2.destroyAllWindows()
            
            # Final status message
            if face_recognized and id_strap_detected:
                self.currentprocess.setText(f"Process complete. Attendance registered for {person_name}.")
            else:
                self.currentprocess.setText("Process complete. No attendance registered.")
            
        except Exception as e:
            error_msg = f"Error in attendance recording: {e}"
            print(error_msg)
            self.currentprocess.setText(error_msg)
            QMessageBox.warning(self, "Error", error_msg)
            self.send_telegram_message(f"ERROR in attendance system: {str(e)}")
            
            # Make sure to release resources
            try:
                webcam.release()
                cv2.destroyAllWindows()
            except:
                pass


    ### SEND TELEGRAM PHOTO ###
    def send_telegram_photo(self, image_path, caption):
        """Sends a photo to Telegram with caption."""
        
        if not os.path.exists(image_path):
            print(f"❌ Error: Image not found at {image_path}")
            self.currentprocess.setText("Error: Failed to send image - file not found")
            return False

        url = f"https://api.telegram.org/bot{self.BOT_TOKEN}/sendPhoto"
        try:
            with open(image_path, "rb") as image_file:
                payload = {"chat_id": self.CHAT_ID, "caption": caption}
                files = {"photo": image_file}
                response = requests.post(url, data=payload, files=files)
                response.raise_for_status()
                print("📩 Telegram API Response:", response.json())
                print("✅ Telegram photo sent successfully!")
                return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Telegram photo failed to send: {e}")
            self.currentprocess.setText(f"Error: Failed to send photo - {str(e)}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error sending Telegram photo: {e}")
            self.currentprocess.setText("Error: Failed to send photo")
            return False

    ### SEND TELEGRAM FILE ###
    def send_telegram_file(self, file_path, message):
        """Sends a file (e.g., .txt) to Telegram."""
        
        if not os.path.exists(file_path):
            print(f"❌ Error: File not found at {file_path}")
            self.currentprocess.setText("Error: Failed to send notification - file not found")
            return False

        url = f"https://api.telegram.org/bot{self.BOT_TOKEN}/sendDocument"
        try:
            with open(file_path, "rb") as file:
                payload = {"chat_id": self.CHAT_ID, "caption": message}
                files = {"document": file}
                response = requests.post(url, data=payload, files=files)
                response.raise_for_status()  # Raise an error for bad status codes
                print("📩 Telegram API Response:", response.json())
                print("✅ Telegram file sent successfully!")
                return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Telegram file failed to send: {e}")
            self.currentprocess.setText(f"Error: Failed to send notification - {str(e)}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error sending Telegram message: {e}")
            self.currentprocess.setText("Error: Failed to send notification")
            return False

    ### SEND TELEGRAM MESSAGE ###
    def send_telegram_message(self, message):
        """Sends a text message to Telegram without a file attachment."""
        
        url = f"https://api.telegram.org/bot{self.BOT_TOKEN}/sendMessage"
        try:
            payload = {"chat_id": self.CHAT_ID, "text": message}
            response = requests.post(url, data=payload)
            response.raise_for_status()
            print("📩 Telegram API Response:", response.json())
            print("✅ Telegram message sent successfully!")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Telegram message failed to send: {e}")
            self.currentprocess.setText(f"Error: Failed to send message - {str(e)}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error sending Telegram message: {e}")
            self.currentprocess.setText("Error: Failed to send message")
            return False

def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()