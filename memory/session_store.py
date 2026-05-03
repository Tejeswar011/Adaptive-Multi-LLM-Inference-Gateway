class SessionStore:

    def __init__(self):
        self.sessions = {}

    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def add_message(self, session_id, role, content):
        session = self.get_session(session_id)
        session.append({"role": role, "content": content})

    def clear_session(self, session_id):
        self.sessions[session_id] = []