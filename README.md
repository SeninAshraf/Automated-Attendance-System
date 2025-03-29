# Automated-Attendance-System
Developed an AI-powered Automated Attendance System using facial recognition (OpenCV, LBPH) and ID card detection (YOLO) for secure, real-time attendance tracking. Prevents proxy attendance, adapts to lighting conditions, and enhances security. Ideal for schools, offices, factories, and hospitals to ensure accurate and efficient attendance logging.

This project introduces an AI-driven attendance system that leverages facial recognition and ID card verification to ensure secure, accurate, and real-time attendance tracking. Using OpenCV, LBPH for face recognition, and YOLO for ID card detection, the system prevents proxy attendance and enhances workplace security.

The system captures and trains facial data, linking each face with an ID for verification before marking attendance. It adapts to different lighting conditions and works efficiently in schools, offices, and industrial settings.

How It Works

1. Face Enrollment: Users register by capturing facial images and linking them with their identity.


2. Real-Time Detection: The camera detects a person’s face and matches it with stored data.


3. ID Card Verification: The system cross-checks the ID card with the recognized face to prevent fraud.


4. Attendance Logging: Upon successful verification, attendance is marked automatically in the database.



Advantages

Prevents Proxy Attendance: Ensures only authorized individuals are marked present.

Fast & Accurate: Works in real-time with high recognition accuracy.

Adaptability: Functions effectively in various lighting and background conditions.

Security Enhancement: Integrates face and ID verification for double authentication.


Technologies Used

OpenCV & LBPH: Facial recognition and feature extraction.

YOLO: ID card detection and validation.

Flask & SQL: Backend processing and attendance storage.


Future Enhancements

Cloud-based Attendance Records: For remote access and real-time monitoring.

Mobile App Integration: Allowing users to check attendance via smartphone.

Voice & Alert System: Notifications for unauthorized attempts.


This AI-powered attendance system ensures efficient workforce management while enhancing security and automation in various industries.
