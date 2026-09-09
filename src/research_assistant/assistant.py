from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .ollama_client import OllamaClient, OllamaError
from .research_tools import ResearchTools, ToolOutput
from .vector_store import FaissVectorStore, SearchHit


@dataclass
class AssistantResponse:
    answer: str
    mode: str
    route_source: str
    elapsed_seconds: float
    hits: list[SearchHit] = field(default_factory=list)
    tool_name: str | None = None
    artifacts: tuple[str, ...] = ()

    @property
    def top_score(self) -> float | None:
        return self.hits[0].score if self.hits else None


class ResearchAssistant:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.client = OllamaClient(
            self.settings.ollama_url,
            llm_model=self.settings.llm_model,
            embedding_model=self.settings.embedding_model,
            timeout=self.settings.request_timeout,
            seed=self.settings.random_seed,
        )
        self.store = FaissVectorStore(self.settings.index_dir, self.client)
        self.tools = ResearchTools(self.settings)

    def _fallback_route(self, query: str) -> str | None:
        lowered = query.casefold()
        patterns: list[tuple[str, tuple[str, ...]]] = [
            ("compare_references", ("reference", "references", "رفرنس", "منابع", "سال انتشار منابع")),
            ("compare_limitations", ("compare limitations", "مقایسه محدودیت", "مقایسه نقاط ضعف")),
            ("find_research_gap", ("research gap", "future work", "شکاف پژوهشی", "کار آینده", "تحقیقات آینده")),
            ("generate_survey", ("survey", "مرور", "جمع بندی چهار مقاله", "جمع‌بندی چهار مقاله")),
            ("compare_results", ("compare result", "compare performance", "best em", "best accuracy", "مقایسه نتایج", "مقایسه عملکرد", "نمودار")),
            ("extract_experimental_results", ("experimental result", "exact match", "em score", "reported result", "achieved", "metric", "dataset", "accuracy", "نتایج تجربی", "معیار", "دیتاست", "امتیاز", "دقت")),
        ]
        for name, needles in patterns:
            if any(needle in lowered for needle in needles):
                return name
        return None

    def route(self, query: str) -> tuple[str | None, dict[str, Any], str]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are only a tool router. If the user explicitly requests one of the available research operations, "
                    "call exactly one matching tool. For a normal question about paper content, do not answer it and reply only RAG."
                ),
            },
            {"role": "user", "content": query},
        ]
        try:
            result = self.client.chat(messages, tools=self.tools.ollama_schemas(), max_tokens=80)
            if result.tool_calls:
                call = result.tool_calls[0].get("function", {})
                name = call.get("name")
                arguments = call.get("arguments", {}) or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments or "{}")
                if name in self.tools.function_names:
                    return name, arguments, "llm_tool_call"
        except (OllamaError, json.JSONDecodeError):
            pass
        fallback = self._fallback_route(query)
        return fallback, {}, "deterministic_fallback" if fallback else "rag"

    def _run_tool(self, name: str, arguments: dict[str, Any], route_source: str, started: float) -> AssistantResponse:
        output: ToolOutput = self.tools.call(name, arguments)
        return AssistantResponse(
            answer=output.text,
            mode="function_call",
            route_source=route_source,
            elapsed_seconds=time.perf_counter() - started,
            tool_name=name,
            artifacts=tuple(str(path) for path in output.artifacts),
        )

    def _context_from_hits(self, hits: list[SearchHit]) -> str:
        blocks = []
        for number, hit in enumerate(hits, start=1):
            meta = hit.metadata
            blocks.append(
                f"[SOURCE {number}: {meta['paper_short_name']}, PDF page {meta['page']}, section {meta['section']}, cosine={hit.score:.3f}]\n"
                f"{meta['text']}"
            )
        return "\n\n".join(blocks)

    def _fallback_general_answer(self, query: str) -> str:
        result = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer the question briefly from general knowledge. Be explicit about uncertainty. "
                        "Do not pretend that the answer came from the selected papers."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=300,
        )
        return (
            "پاسخ شما در مقالات انتخاب‌شده یافت نشد.\n"
            "اما با استفاده از مدل زبانی، پاسخ عمومی به شرح زیر است:\n"
            f"{result.content}"
        )

    def ask(self, query: str, *, allow_tools: bool = True) -> AssistantResponse:
        query = query.strip()
        if not query:
            raise ValueError("Question cannot be empty.")
        started = time.perf_counter()

        if allow_tools:
            tool_name, arguments, route_source = self.route(query)
            if tool_name:
                return self._run_tool(tool_name, arguments, route_source, started)
        else:
            route_source = "tools_disabled"

        hits = self.store.search(query, top_k=self.settings.retrieval_top_k)
        if not hits or hits[0].score < self.settings.relevance_threshold:
            answer = self._fallback_general_answer(query)
            return AssistantResponse(
                answer=answer,
                mode="general_fallback",
                route_source=route_source,
                elapsed_seconds=time.perf_counter() - started,
                hits=hits,
            )

        context = self._context_from_hits(hits)
        prompt = (
            f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\n"
            "Give a concise answer using only sentences that directly answer the question. Do not mix a retriever description with a generator description. "
            "Do not infer missing details. Cite sources as [DPR, p. 8] or the corresponding short name/page. "
            "If no excerpt contains the answer, reply NO_ANSWER_IN_CONTEXT."
        )
        result = self.client.chat(
            [
                {"role": "system", "content": "Use only the supplied context and do not invent facts."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=450,
        )
        if re.search(r"\bNO_ANSWER_IN_CONTEXT\b", result.content, re.IGNORECASE):
            answer = self._fallback_general_answer(query)
            mode = "general_fallback"
        else:
            answer = result.content
            allowed = {
                (hit.metadata["paper_short_name"].casefold(), int(hit.metadata["page"]))
                for hit in hits
            }
            citation_pattern = re.compile(
                r"\[(REALM|DPR|RAG|FiD),\s*p\.\s*(\d+)\]", re.IGNORECASE
            )

            def validate_citation(match: re.Match[str]) -> str:
                key = (match.group(1).casefold(), int(match.group(2)))
                return match.group(0) if key in allowed else ""

            answer = citation_pattern.sub(validate_citation, answer).strip()
            if not citation_pattern.search(answer):
                citations = []
                for hit in hits[:2]:
                    citation = hit.citation
                    if citation not in citations:
                        citations.append(citation)
                answer += "\n\nRetrieved evidence: " + " ".join(citations)
            mode = "rag"
        return AssistantResponse(
            answer=answer,
            mode=mode,
            route_source=route_source,
            elapsed_seconds=time.perf_counter() - started,
            hits=hits,
        )

    def answer_without_rag(self, query: str) -> str:
        return self.client.chat(
            [
                {"role": "system", "content": "Answer briefly. If unsure, say so."},
                {"role": "user", "content": query},
            ],
            max_tokens=180,
        ).content
