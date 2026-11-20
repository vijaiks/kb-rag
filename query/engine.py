"""Conversational RAG engine shared by the terminal chat and the local web UI.

One ChatEngine = one conversation. Per turn:
  condense follow-up -> retrieve (hybrid + rerank) -> messages (history + fresh sources)
  -> stream the answer -> remember the turn.

History stores only the plain question/answer text, never old chunks, so context stays small.
Source tags [S1..] restart each turn and always refer to the latest Sources block.
"""
from __future__ import annotations

from collections.abc import Iterator

from common import chunks_table, embed_texts, load_config, tech_filter
from llm.client import LLMClient
from query.ask import SYSTEM as KB_SYSTEM
from query.ask import build_prompt, retrieve

MODES = ("kb", "assistant")

HISTORY_NOTE = (
    "\nThe conversation may contain earlier turns. Citation tags in earlier answers refer to "
    "earlier sources; cite only tags from the Sources block of the latest message."
)

ASSISTANT_SYSTEM = (
    "You are a telecom and software architecture assistant with access to the user's personal "
    "knowledge base.\n"
    "Rules:\n"
    "- Prefer the provided sources. Cite each claim taken from them with a tag like [S2].\n"
    "- You may add general knowledge when the sources are silent, but mark it clearly, e.g. "
    "'(general knowledge, not from your KB)', and never attach a [S#] tag to it.\n"
    "- Be concise and use plain language; expand acronyms once."
)

CONDENSE_SYSTEM = (
    "Rewrite the user's latest message as one standalone search query, resolving pronouns and "
    "references using the conversation. Output only the query, nothing else."
)

NOT_FOUND = "Not found in the knowledge base."


class ChatEngine:
    def __init__(self, cfg: dict | None = None, llm=None, tbl=None, retriever=retrieve,
                 max_turns: int = 6):
        self.cfg = cfg or load_config()
        self.tbl = tbl if tbl is not None else chunks_table(self.cfg)
        self.llm = llm or LLMClient(self.cfg)
        self.retriever = retriever
        self.max_turns = max_turns
        self.history: list[dict] = []  # alternating user/assistant plain-text messages
        self.mode = "kb"
        self.fast = False
        self._reranker = self.cfg["retrieval"].get("reranker")  # restored when fast is turned off

    # ---- settings -------------------------------------------------------
    def warm_up(self) -> None:
        embed_texts(self.cfg, ["warm up"])  # loads BGE-M3 once

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode

    def set_fast(self, fast: bool) -> None:
        self.fast = fast
        self.cfg["retrieval"]["reranker"] = None if fast else self._reranker

    def reset(self) -> None:
        self.history.clear()

    # ---- one turn ---------------------------------------------------------
    def condense(self, question: str) -> str:
        """Standalone retrieval query for a follow-up. First turn / failures: the question itself."""
        if not self.history:
            return question
        convo = "\n".join(f"{m['role']}: {m['content'][:600]}" for m in self.history[-4:])
        try:
            out = self.llm.complete(
                CONDENSE_SYSTEM, f"Conversation:\n{convo}\n\nLatest message: {question}", max_tokens=100
            ).strip()
        except Exception:  # noqa: BLE001 - retrieval with the raw question still works
            return question
        return out or question

    def reply(self, question: str, tech: str | None = None) -> Iterator[dict]:
        """Yield events: {"type":"query","text"}, {"type":"sources","sources":[...]},
        {"type":"token","text"}. The turn enters the history only once fully streamed."""
        query = self.condense(question)
        yield {"type": "query", "text": query}
        chunks = self.retriever(self.cfg, self.tbl, query, tech_filter(tech))
        yield {"type": "sources", "sources": [
            {"tag": f"S{i}", "file": c["file_path"], "page_start": c["page_start"],
             "page_end": c["page_end"], "section": c["section_path"], "text": c["text"]}
            for i, c in enumerate(chunks, 1)
        ]}

        if not chunks and self.mode == "kb":
            answer = NOT_FOUND
            yield {"type": "token", "text": answer}
        else:
            user = build_prompt(question, chunks) if chunks else question
            system = (KB_SYSTEM if self.mode == "kb" else ASSISTANT_SYSTEM) + (
                HISTORY_NOTE if self.history else "")
            parts = []
            for piece in self.llm.stream(system, self.history + [{"role": "user", "content": user}],
                                         max_tokens=700 if self.fast else None):
                parts.append(piece)
                yield {"type": "token", "text": piece}
            answer = "".join(parts)

        self.history += [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}]
        del self.history[: -2 * self.max_turns]
