# GoodMem for LangChain4j

Use [GoodMem](https://goodmem.ai) as the memory and retrieval service for your Java application. Write LangChain4j Documents, retrieve text with source metadata, and connect it to your existing agent or RAG workflow. GoodMem handles chunking, embedding, storage and optional reranking.

## Install

Requires **Java 21+** and LangChain4j 1.20.0+.

```xml
<dependency>
    <groupId>io.github.bashareid</groupId>
    <artifactId>goodmem-langchain4j</artifactId>
    <version>0.2.0</version>
</dependency>
```

The integration includes the official GoodMem Java SDK and LangChain4j. For AI Services, add your preferred chat-model provider.

## Store and retrieve documents

[Set up GoodMem](https://docs.goodmem.ai/docs/how-to/basic-rag/) and a space, then set `GOODMEM_BASE_URL`, `GOODMEM_API_KEY` and `GOODMEM_SPACE_ID`. Optionally set `GOODMEM_RERANKER_ID` to enable reranking; it does not require an LLM.

```java
import ai.pairsys.goodmem.client.Goodmem;
import ai.pairsys.goodmem.langchain4j.GoodMemContentRetriever;
import ai.pairsys.goodmem.langchain4j.GoodMemDocumentIngestor;
import ai.pairsys.goodmem.langchain4j.GoodMemFilters;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.rag.query.Query;
import java.time.Duration;
import java.util.List;
import java.util.Map;

public class Quickstart {
    public static void main(String[] args) {
        String spaceId = System.getenv("GOODMEM_SPACE_ID");
        try (Goodmem client = Goodmem.builder()
                .baseUrl(System.getenv("GOODMEM_BASE_URL"))
                .apiKey(System.getenv("GOODMEM_API_KEY"))
                .build()) {
            var ingestor = GoodMemDocumentIngestor.builder()
                    .client(client).spaceId(spaceId).build();
            ingestor.ingestAndWait(List.of(Document.from(
                    "Customers may return unused items within 30 days.",
                    Metadata.from(Map.of("source", "handbook", "team", "support")))), Duration.ofMinutes(2));

            var retriever = GoodMemContentRetriever.builder()
                    .client(client).spaceIds(List.of(spaceId))
                    .filterExpression(GoodMemFilters.textEquals("team", "support"))
                    .rerankerId(System.getenv("GOODMEM_RERANKER_ID"))
                    .build();
            for (var content : retriever.retrieve(Query.from("What is the return policy?"))) {
                System.out.println(content.textSegment());
            }
        }
    }
}
```

`ingestAndWait` returns memory IDs after indexing, within the timeout you choose. Use `ingest` to return as soon as writes are accepted. Readiness failures retain the IDs so you can wait again without uploading again. Each ingestion creates new memories.

## Connect your agent

Pass the retriever to `AiServices.builder(...).contentRetriever(retriever)`. A `Result<String>` return type gives you both the answer and `sources()`.

For agent-directed search, use `retriever.asTool("searchPolicies", "Search our customer policies")`. Register a `List<AiServiceTool>` through `.tools(searchTools)`. Its only input is the query; your application configures the spaces, filters and reranker. Agents that need to manage spaces or write memories can use `new GoodMemTools(client)`. Local file uploads require an explicitly configured directory.

See the [usage guide](docs/usage.md) for AI Services, dynamic filters, async workflows and tool results, and the [migration notes](CHANGELOG.md) before upgrading from 0.1.

## Development

```bash
./mvnw verify
./mvnw spotless:apply
```

The tests exercise the real SDK over a local HTTP fixture and use LangChain4j's actual tool and RAG APIs. [Live tests](docs/usage.md#live-tests) additionally check a running GoodMem server. The README example is compiled during testing.

[MIT](LICENSE) · [Source](https://github.com/PAIR-Systems-Inc/goodmem-langchain4j) · [Maven Central](https://central.sonatype.com/artifact/io.github.bashareid/goodmem-langchain4j)
