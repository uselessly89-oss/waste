from collections import Counter, deque


class TemporalVoter:
    """Track-local consensus to prevent single-frame class flips."""

    def __init__(self, window=10):
        self.window = window
        self.history = {}

    def update(self, track_id, class_id, confidence):
        q = self.history.setdefault(track_id, deque(maxlen=self.window))
        q.append((class_id, confidence))
        counts = Counter(c for c, _ in q)
        winner, votes = counts.most_common(1)[0]
        ratio = votes / len(q)
        winner_conf = sum(c for cls, c in q if cls == winner) / votes
        return winner, winner_conf, ratio, len(q)

    def clear(self, track_id):
        self.history.pop(track_id, None)
