memory_db = {}

def get_user_context(user_id):
    return memory_db.get(user_id, "")

def memory_store(user_id, user_input, bot_reply):
    if user_id not in memory_db:
        memory_db[user_id] = ""
    memory_db[user_id] += f"\nUser: {user_input}\nAssistant: {bot_reply}"
