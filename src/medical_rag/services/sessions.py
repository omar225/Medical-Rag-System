"""Bounded in-memory conversation history, keyed by the client's session_id.

Only the last few turns are kept so follow-ups ("what about its contraindications?") can be
resolved. History is lost on restart by design: no patient data is persisted.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass
from threading import Lock
from time import monotonic


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str


class SessionStore:
    def __init__(self, max_sessions: int = 1000, max_turns: int = 3, ttl_seconds: float = 3600):
        self.max_sessions, self.max_turns, self.ttl = max_sessions, max_turns, ttl_seconds
        self._sessions: OrderedDict[str, tuple[float, deque]] = OrderedDict()
        self._lock = Lock()

    def history(self, session_id: str | None) -> tuple[Turn, ...]:
        if not session_id:
            return ()
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None or monotonic() - entry[0] > self.ttl:
                self._sessions.pop(session_id, None)
                return ()
            return tuple(entry[1])

    def append(self, session_id: str | None, turn: Turn) -> None:
        if not session_id:
            return
        with self._lock:
            _, turns = self._sessions.pop(session_id, (0.0, deque(maxlen=self.max_turns)))
            turns.append(turn)
            self._sessions[session_id] = (monotonic(), turns)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
