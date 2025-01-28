import cv2
import mediapipe as mp


# Function to detect poses
def detect_pose(frame, pose_detector):
    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect poses in the image
    results = pose_detector.process(image_rgb)

    # Draw pose landmarks on the image if poses are detected
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)


# Main function
def main():
    # OpenCV video capture
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

    # Initialize Mediapipe Pose model
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose()

    while cap.isOpened():
        # Read frame from webcam
        success, frame = cap.read()
        if not success:
            break

        # Detect pose in the frame
        detect_pose(frame, pose_detector)

        # Display the frame
        cv2.imshow('Pose Detection', frame)

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()