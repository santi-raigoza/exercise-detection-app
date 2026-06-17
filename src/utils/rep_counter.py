from collections import deque

import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

SMOOTH_N = 5
CONFIDENCE_THRESHOLD = 0.7


def _angle(a, b, c):
    """Angle in degrees at point b, formed by vectors b→a and b→c."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


class RepCounter:
    def __init__(self):
        self._exercise = None
        self._count = 0
        self._state = None
        self._history = deque(maxlen=SMOOTH_N)

    def _metric(self, exercise, landmarks):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        vis = np.array([getattr(lm, 'visibility', 0.5) for lm in landmarks], dtype=np.float32)

        if exercise == 'pushup':
            left = _angle(coords[LEFT_SHOULDER], coords[LEFT_ELBOW], coords[LEFT_WRIST])
            right = _angle(coords[RIGHT_SHOULDER], coords[RIGHT_ELBOW], coords[RIGHT_WRIST])
            left_vis = (vis[LEFT_SHOULDER] + vis[LEFT_ELBOW] + vis[LEFT_WRIST]) / 3
            right_vis = (vis[RIGHT_SHOULDER] + vis[RIGHT_ELBOW] + vis[RIGHT_WRIST]) / 3
            return left if left_vis >= right_vis else right
        if exercise == 'pullup':
            avg_shoulder_y = (coords[LEFT_SHOULDER][1] + coords[RIGHT_SHOULDER][1]) / 2
            avg_wrist_y = (coords[LEFT_WRIST][1] + coords[RIGHT_WRIST][1]) / 2
            return avg_shoulder_y - avg_wrist_y
        if exercise == 'squat':
            left = _angle(coords[LEFT_HIP], coords[LEFT_KNEE], coords[LEFT_ANKLE])
            right = _angle(coords[RIGHT_HIP], coords[RIGHT_KNEE], coords[RIGHT_ANKLE])
            left_vis = (vis[LEFT_HIP] + vis[LEFT_KNEE] + vis[LEFT_ANKLE]) / 3
            right_vis = (vis[RIGHT_HIP] + vis[RIGHT_KNEE] + vis[RIGHT_ANKLE]) / 3
            return left if left_vis >= right_vis else right
        if exercise == 'jumping_jack':
            shoulder_width = abs(coords[LEFT_SHOULDER][0] - coords[RIGHT_SHOULDER][0]) + 1e-6
            wrist_spread = abs(coords[LEFT_WRIST][0] - coords[RIGHT_WRIST][0]) / shoulder_width
            ankle_spread = abs(coords[LEFT_ANKLE][0] - coords[RIGHT_ANKLE][0]) / shoulder_width
            return min(wrist_spread, ankle_spread)
        return 0.0

    def update(self, exercise, confidence, landmarks):
        """
        exercise:   predicted class string
        confidence: float in [0, 1]
        landmarks:  list of 33 objects with .x .y .z
        Returns:    current rep count (int)
        """
        if exercise == 'rest' or confidence < CONFIDENCE_THRESHOLD:
            return self._count

        if exercise != self._exercise:
            self._exercise = exercise
            self._count = 0
            self._state = 'up' if exercise in ('pushup', 'squat') else 'down'
            self._history.clear()

        self._history.append(self._metric(exercise, landmarks))
        smoothed = float(np.mean(self._history))

        if exercise == 'pushup':
            if self._state == 'up' and smoothed < 110:
                self._state = 'down'
            elif self._state == 'down' and smoothed > 130:
                self._state = 'up'
                self._count += 1

        elif exercise == 'squat':
            if self._state == 'up' and smoothed < 110:
                self._state = 'down'
            elif self._state == 'down' and smoothed > 140:
                self._state = 'up'
                self._count += 1

        elif exercise == 'pullup':
            if self._state == 'down' and smoothed > 0.15:
                self._state = 'up'
            elif self._state == 'up' and smoothed < 0.08:
                self._state = 'down'
                self._count += 1

        elif exercise == 'jumping_jack':
            if self._state == 'down' and smoothed > 1.3:
                self._state = 'up'
            elif self._state == 'up' and smoothed < 0.7:
                self._state = 'down'
                self._count += 1

        return self._count

    @property
    def count(self):
        return self._count
