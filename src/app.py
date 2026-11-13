import time

import cv2

from pipeline import ExerciseSession, create_landmarker, load_classifier


def main():
    model, scaler = load_classifier()
    landmarker = create_landmarker()
    cap = cv2.VideoCapture(0)

    session = ExerciseSession(model, scaler)
    prev_time = time.time()

    print('Press Q to quit.')
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        session.process(frame, landmarker, t=now, fps=fps)
        cv2.imshow('Exercise Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == '__main__':
    main()
