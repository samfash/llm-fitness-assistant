"""Tiny in-memory chat memory (resets when the Space restarts)."""
MAX_TURNS = 6  # only the last N exchanges go into the prompt so it never outgrows the context window

memory_db = {}


def get_user_context(user_id):
    turns = memory_db.get(user_id, [])[-MAX_TURNS:]
    return "\n".join(f"User: {u}\nAssistant: {a}" for u, a in turns)


def memory_store(user_id, user_input, bot_reply):
    memory_db.setdefault(user_id, []).append((user_input, bot_reply))
