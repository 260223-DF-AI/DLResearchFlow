"""
ResearchFlow — Cross-Thread Memory (Store Interface)

Manages user preferences and query history across threads
using the LangGraph Store interface with namespaces and scopes.
"""

from langgraph.store.memory import InMemoryStore

# Module-level singleton — one Store across the whole process.
# In Lambda you would swap this for PostgresStore so memory survives
# between invocations.
_store = InMemoryStore()

DEFAULT_PREFERENCES = {
    "verbosity": "normal",         # "concise" | "normal" | "verbose"
    "trusted_sources": [],
}


def get_user_preferences(user_id: str) -> dict:
    """Read prefs for a user, or return defaults if absent."""
    namespace = ("users", user_id)
    item = _store.get(namespace, "preferences")
    return item.value if item else dict(DEFAULT_PREFERENCES)


def save_user_preferences(user_id: str, preferences: dict) -> None:
    """Overwrite the preferences blob for this user."""
    _store.put(("users", user_id), "preferences", preferences)


def get_query_history(user_id: str, limit: int = 5) -> list[str]:
    """Return the most recent N queries this user has asked.

    History is stored as a single list under one key — fine for a few
    thousand entries; switch to per-query keys + a search index past that.
    """
    item = _store.get(("users", user_id, "history"), "queries")
    if not item:
        return []
    return item.value[-limit:]


def append_query(user_id: str, question: str) -> None:
    """Append `question` to this user's history."""
    namespace = ("users", user_id, "history")
    item = _store.get(namespace, "queries")
    history = item.value if item else []
    history.append(question)
    _store.put(namespace, "queries", history)


if __name__ == "__main__":
    save_user_preferences("alice", {"verbosity": "verbose", "trusted_sources": ["nytimes.com"]})
    print(get_user_preferences("alice"))
    # → {'verbosity': 'verbose', 'trusted_sources': ['nytimes.com']}
    
    append_query("alice", "What is the GDP of France?")
    append_query("alice", "What is the population of France?")
    print(get_query_history("alice"))
    # → ['What is the GDP of France?', 'What is the population of France?']
    
    print(get_user_preferences("bob"))
    # → defaults
