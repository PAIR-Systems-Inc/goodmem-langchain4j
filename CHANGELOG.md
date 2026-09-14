# Changelog

## 0.2.0 (2026-09-14)

This is a clean API break. Requires Java 21 and LangChain4j 1.20.0. Maven coordinates remain `io.github.bashareid:goodmem-langchain4j`.

- Replaced the hand-written HTTP client with `ai.pairsys:goodmem-java:0.2.2`. Future server status codes map to `UNKNOWN` without breaking retrieval; messages and details are preserved.
- Added `GoodMemContentRetriever`, implementing LangChain4j's standard `ContentRetriever` with source metadata, SDK IDs, metadata filters, per-query filters, reranking and framework listener support.
- Added `GoodMemDocumentIngestor` for whole Documents and metadata, batched writes and confirmed IDs after partial failure. `ingest` returns accepted IDs; `ingestAndWait(documents, timeout)` opts into readiness with an explicit budget.
- Added `retriever.asTool(name, description)`, returning a native `AiServiceTool`. Multiple scoped searches can coexist in one agent; only query text is model-visible.
- Added `GoodMemIndexing.waitForMemory` and batched `waitForMemories` for specific written IDs. `GoodMemIndexingException` retains all/pending IDs so readiness can be retried without another upload.
- Removed empty-search polling. The create-memory agent tool waits for its own memory by default.
- Retrieval tools return compact chunks, sources, scores, optional `abstractReply`, statuses and `partial`. Non-informational statuses, including `UNKNOWN`, mark partial results without discarding useful chunks. The content retriever throws for known non-informational statuses while tolerating future unknown codes. Protocol and HTTP failures still propagate.
- `fetchK` controls candidates with or without a post-processor. Metadata collisions preserve both original maps in `goodmem_metadata` JSON while keeping canonical IDs and convenient flat fields.
- Native filters use the explicit `filterExpression` and `dynamicFilterExpression` methods. `GoodMemFilters.textEquals` safely constructs a text comparison from a field name and value.
- Space listing follows SDK pagination. Creation always creates a new space using SDK chunking defaults; it does not reuse spaces by name. Removed chunking controls and `publicRead` from tool inputs.
- Get-memory returns readable text by default, with charset-aware decoding and a notice for omitted binary content. Original bytes remain available through the SDK.
- Local file uploads require an explicitly configured upload directory; paths and symlinks escaping it are rejected.
- SDK errors before confirmed writes preserve their types. Partial-write exceptions retain the original cause alongside confirmed IDs; readiness failures use a separate exception.
- Added SDK transport tests, LangChain4j tool/RAG/listener tests, compiled documentation examples, live tests, Spotless and Checkstyle checks, and corrected CI artifact paths.
- Version tags now publish signed artifacts to Maven Central and wait for publication to finish.

### Migrating callers

| 0.1 | 0.2 |
| --- | --- |
| `GoodMemClient` | Official `Goodmem` SDK client |
| `GoodMemTools.builder().baseUrl(...).apiKey(...).build()` | `new GoodMemTools(client)`; the caller creates and closes the SDK client |
| Tool JSON strings with `success`, `results` and counts | Compact retrieval/read records and SDK administrative models, serialized by LangChain4j; failures throw exceptions |
| Comma-separated space IDs | `List<String>` |
| JSON strings for metadata and labels | Java maps |
| `waitForIndexing` on retrieval | Removed; configure write readiness instead |
| Same-name space reuse and chunking arguments | Explicit space IDs for reuse; use SDK requests for custom chunking |
| File input silently overrides text; unrestricted local file access | Provide exactly one input; uploads require a configured directory |
| Post-processing temperature, threshold and chronological-sort tool arguments | Configure advanced post-processing directly through the SDK |
| Manual pagination tokens | Listing tools iterate pages, capped at `maxItems` (100 by default); use SDK `Page` for cursor control |
| `.chatLanguageModel(model)` in examples | `.chatModel(model)` |

For native search tools, pass a typed `List<AiServiceTool>` to `AiServices.builder(...).tools(searchTools)`. Use a unique name and useful source description for each scoped retriever.

The SDK's async retrieval exposes a blocking stream iterator. For async RAG, use LangChain4j's explicit `DefaultRetrievalAugmentor.offloadBlocking(true)` support. The integration does not claim native nonblocking streaming.

## 0.1.0

Initial release of eleven JSON-returning LangChain4j tools, published May 5, 2026.
