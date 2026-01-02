import cv2
import mediapipe as mp
import joblib

model = joblib.load("model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        features = []
        for lm in landmarks.landmark:
            features.extend([lm.x, lm.y])

        prediction = model.predict([features])[0]
        cv2.putText(frame, prediction, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
