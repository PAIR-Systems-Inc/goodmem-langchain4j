# Using GoodMem with LangChain4j

Create a `Goodmem` SDK client with your endpoint and credentials, then pass it to the integration. Your application owns the client and closes it when finished. Transport configuration, including a custom OkHttp client, belongs to the SDK.

## AI Services and sources

The Maven coordinate remains under the maintainer's personal `io.github.bashareid` groupId.

LangChain4j and the GoodMem SDK are included; add your chat provider. With an initialized `ChatModel model` and `GoodMemContentRetriever retriever`:

```java
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Result;

interface Assistant {
    Result<String> chat(String question);
}

Assistant assistant = AiServices.builder(Assistant.class)
        .chatModel(model)
        .contentRetriever(retriever)
        .build();
Result<String> answer = assistant.chat("What is the return policy?");
System.out.println(answer.content());
answer.sources().forEach(source -> System.out.println(source.textSegment().metadata()));
```

Each `TextSegment` includes `memory_id`, `chunk_id` and `space_id`, plus stored metadata. A source reference is used when `source` metadata is absent. If neither exists, the memory and chunk IDs still identify the result.

Chunk metadata takes precedence over memory metadata in the flat fields; the three IDs always identify the actual server records. When names collide, `goodmem_metadata` contains both untouched original maps as JSON, under `memory` and `chunk`. This also preserves any user field named `goodmem_metadata`.

LangChain4j metadata supports strings, UUIDs and numeric scalars. Other GoodMem metadata values, including nested objects, lists and booleans, are represented as JSON strings; null values are omitted. Scores are available through `content.metadata().get(ContentMetadata.SCORE)`. Reranked results also carry `RERANKED_SCORE`. Server order and score values are preserved; scores should not be treated as a universal 0-to-1 scale.

To include source names in the model's prompt, configure a `DefaultContentInjector` with `metadataKeysToInclude(List.of("source"))` on a `DefaultRetrievalAugmentor`. The `Result.sources()` API remains available independently of prompt formatting.

For an agent that decides when to search, register a named, described tool:

```java
interface SearchAssistant {
    String chat(String question);
}

var searchTools = List.of(retriever.asTool("searchPolicies", "Search our customer return policies"));
var assistant = AiServices.builder(SearchAssistant.class)
        .chatModel(model)
        .tools(searchTools)
        .build();
```

The list is typed as `List<AiServiceTool>`, which selects LangChain4j's native tool overload. Add separately configured retrievers to this list with unique names and descriptions, for example policies and tickets. Each tool exposes only a query; its spaces, filters and reranker remain fixed by the application.

## Filters and reranking

`filterExpression(String)` applies a native GoodMem expression to each configured space. For example:

```java
import ai.pairsys.goodmem.langchain4j.GoodMemFilters;

var retriever = GoodMemContentRetriever.builder()
        .client(client)
        .spaceIds(List.of(spaceId))
        .filterExpression(GoodMemFilters.textEquals("team", "support"))
        .rerankerId(rerankerId)
        .maxResults(5)
        .fetchK(20)
        .build();
```

`dynamicFilterExpression(Function<Query, String>)` adds a per-request expression, combined with the fixed expression using `AND`. These methods take native GoodMem expressions, not LangChain4j `Filter` objects. Use `GoodMemFilters.textEquals("tenant", tenant)` for text values from application context; it escapes quotes, backslashes and line breaks. Field names must be simple top-level identifiers. For more elaborate native expressions, keep the expression application-controlled and avoid inserting unescaped user text.

Reranking uses the configured GoodMem reranker without requiring an LLM. `fetchK` controls the candidate count; `maxResults` caps returned content. Defaults are five returned results and twenty candidates when reranking is enabled.

## Document ingestion and indexing

`GoodMemDocumentIngestor.ingest(List<Document>)` uploads whole documents in batches of at most 100 and returns accepted memory IDs in input order. GoodMem owns chunking and embedding; no local embedding model is needed.

Use `ingestAndWait(documents, Duration.ofMinutes(5))` when the next operation needs indexed memories. You choose the polling budget explicitly. Readiness uses batch status reads for the IDs just written and stops polling completed IDs. It never retries empty searches. SDK request timeouts apply separately to each HTTP call.

If readiness times out or fails, `GoodMemIndexingException.memoryIds()` retains all accepted IDs and `pendingMemoryIds()` identifies those still unconfirmed. Resume with `GoodMemIndexing.waitForMemories(client, pendingIds, timeout)` without rewriting the documents. The same method supports SDK-written batches; `waitForMemory` is available for a single ID.

Each ingestion creates new memories. Source downloading, deduplication and replacement policies belong to the application. A failed batch is not rolled back or automatically retried. An SDK failure before any confirmed writes keeps its original exception type, such as `AuthenticationException`. After confirmed writes, `GoodMemIngestionException.createdMemoryIds()` retains their IDs and `getCause()` retains the original exception. A transport failure can leave additional writes unconfirmed.

## Administrative tools and diagnostics

`new GoodMemTools(client)` provides space and memory operations as eleven `@Tool` methods. LangChain4j serializes the results and handles exceptions. Direct tool HTTP failures retain their SDK exception types. Configure which tools your agent receives according to its job; use the scoped search tool for a read-only assistant.

`goodmemRetrieveMemories` returns `chunks` (text, source, score and IDs), optional `abstractReply`, `statuses` and `partial`. Any non-informational status sets `partial=true`; useful chunks remain available. The tool description tells the model to acknowledge partial results. `fetchK` controls server candidates even without reranking or summarization. Use the SDK directly when you need the complete event stream.

The content retriever and its scoped search tool throw `GoodMemRetrievalException` for known non-informational statuses; diagnostics are available through `statuses()`. Future server codes map to `UNKNOWN` in the SDK and do not abort content retrieval. They remain visible in the administrative retrieval tool, which marks the result partial. Malformed streams and HTTP failures still throw.

`goodmemGetMemory` returns readable original text by default, respecting its charset. Binary or undecodable content is omitted with `contentNotice`; use the SDK for original bytes. Pass `includeContent=false` for metadata alone. Listing tools traverse pages up to `maxItems`, defaulting to 100; use the SDK for cursor control.

Create-memory waits for that specific memory by default, so an agent can immediately retrieve a note it saved. Pass `wait=false` to return after acceptance. `new GoodMemTools(client, Duration.ofMinutes(2))` configures the per-write wait. Space creation always creates a new space with SDK defaults; duplicate names for the same owner produce the SDK's HTTP 409 `ConflictException`.

Local file uploads are disabled by default. To enable them, use `new GoodMemTools(client, Duration.ofMinutes(2), Path.of("/srv/agent-uploads"))` with an existing directory containing files the agent may read. Relative paths resolve there; absolute paths, traversal and symlinks that escape that directory are rejected. Text ingestion does not require filesystem access.

## Async RAG

The SDK's async retrieval returns a blocking stream iterator. Use LangChain4j's existing offload option for asynchronous AI Services:

```java
var augmentor = DefaultRetrievalAugmentor.builder()
        .contentRetriever(retriever)
        .offloadBlocking(true)
        .build();
```

Pass this to `AiServices.builder(...).retrievalAugmentor(augmentor)`. LangChain4j uses its executor for the blocking stage. This is offloaded retrieval, not native nonblocking streaming.

## Local TLS

For a self-signed development server, obtain its certificate or local CA certificate and add it
to a copy of your JDK truststore. Copying the existing store preserves trust in public services
your application may also call.

```bash
cp "$JAVA_HOME/lib/security/cacerts" ./goodmem-truststore
keytool -importcert -alias goodmem-local -file local-ca.pem -keystore ./goodmem-truststore
java -Djavax.net.ssl.trustStore=./goodmem-truststore \
     -Djavax.net.ssl.trustStorePassword=changeit -jar your-app.jar
```

Use your truststore's password; `changeit` is the usual JDK default. The server certificate must
also match the hostname in `GOODMEM_BASE_URL`. Version 0.2 has no `verifySsl` flag. Standard JVM
TLS settings apply to the SDK's default client; applications using a custom OkHttp client configure
TLS on that client.

## Live tests

Run the ordinary transport and framework tests with `./mvnw verify`. To additionally test a real server, configure:

```bash
export GOODMEM_BASE_URL=http://localhost:8080
export GOODMEM_API_KEY=your-key
export GOODMEM_EMBEDDER_ID=your-embedder-uuid
export GOODMEM_RERANKER_ID=your-reranker-uuid
./mvnw -Plive verify
```

Live tests use the server's configured embedding and reranking providers. They create uniquely named spaces and delete their own spaces afterwards. They exercise ingestion, immediate retrieval after writes, filtering, source metadata, reranking, file uploads and invalid-reranker diagnostics. No GoodMem LLM is required. The ordinary framework tests use a deterministic chat model to verify AI Service wiring and source access.

Build sources and Javadoc artifacts with `./mvnw -Prelease -Dgpg.skip=true verify`. The publish workflow verifies a matching version tag, publishes the signed artifacts to Maven Central and waits for publication to finish.
