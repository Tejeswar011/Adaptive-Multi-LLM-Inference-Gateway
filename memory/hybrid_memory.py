from memory.session_store import SessionStore
from utils.token_counter import estimate_tokens


class HybridMemory:

    def __init__(self, turn_limit=8, token_limit=1000):
        self.store = SessionStore()
        self.turn_limit = turn_limit
        self.token_limit = token_limit

    def get_context(self, session_id):
        session = self.store.get_session(session_id)

        # Sliding window first
        context = session[-self.turn_limit:]

        # Token-based compression check
        total_tokens = sum(estimate_tokens(msg["content"]) for msg in context)

        if total_tokens > self.token_limit:
            # Summarize older half
            midpoint = len(context) // 2
            summary_text = self._simple_summary(context[:midpoint])

            # Replace older half with summary
            context = [
                {"role": "system", "content": f"Summary: {summary_text}"}
            ] + context[midpoint:]

        return context

    def add_turn(self, session_id, role, content):
        self.store.add_message(session_id, role, content)

    def _simple_summary(self, messages):
        # Simple join-based compression for now
        combined = " ".join([m["content"] for m in messages])
        return combined[:500]  # truncate