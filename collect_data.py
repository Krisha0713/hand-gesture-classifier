import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

label = input("Enter gesture label: ")
data = []

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        features = []
        for lm in landmarks.landmark:
            features.extend([lm.x, lm.y])
        data.append(features)

    cv2.imshow("Collecting Data", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

with open("gesture_data.csv", "a", newline="") as f:
    writer = csv.writer(f)
    for row in data:
        writer.writerow(row + [label])

print("Data saved")