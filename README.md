# goodmem-langchain4j

[![Maven Central](https://img.shields.io/maven-central/v/io.github.pair-systems-inc/goodmem-langchain4j.svg)](https://central.sonatype.com/artifact/io.github.pair-systems-inc/goodmem-langchain4j)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A GoodMem connector for [LangChain4j](https://github.com/langchain4j/langchain4j). Store and retrieve memories from a [GoodMem](https://goodmem.ai) server without having to configure your own data processing pipeline.

## What is GoodMem?

GoodMem is a memory layer for AI agents with first-class support for semantic storage, retrieval, and summarization. This package exposes GoodMem operations as LangChain4j `@Tool`s that any LangChain4j agent can call.

## Installation

Add the dependency to your `pom.xml`:

```xml
<dependency>
    <groupId>io.github.pair-systems-inc</groupId>
    <artifactId>goodmem-langchain4j</artifactId>
    <version>0.1.0</version>
</dependency>
```

Requires Java 17+ and `dev.langchain4j:langchain4j-core` 1.14.0 or newer.

## Quickstart

```java
import ai.pairsys.goodmem.langchain4j.GoodMemTools;
import dev.langchain4j.service.AiServices;

GoodMemTools goodMemTools = GoodMemTools.builder()
        .baseUrl("https://localhost:8080")
        .apiKey("your-api-key")
        .verifySsl(false)
        .build();

interface Assistant {
    String chat(String message);
}

Assistant assistant = AiServices.builder(Assistant.class)
        .chatLanguageModel(model)
        .tools(goodMemTools)
        .build();
```

## Configuration

`GoodMemTools.builder()` accepts:

| Option | Description |
|---|---|
| `baseUrl` | URL of the GoodMem server (e.g. `https://localhost:8080`) |
| `apiKey` | API key for authentication |
| `verifySsl` | Disable to accept self-signed certificates in development |

## API Reference

| Tool | Description |
|---|---|
| `goodmemListEmbedders` | List available embedder models |
| `goodmemListSpaces` | List all spaces in your account |
| `goodmemGetSpace` | Fetch a specific space by ID |
| `goodmemCreateSpace` | Create a new space or reuse an existing one |
| `goodmemUpdateSpace` | Update name, public-read flag, or labels of a space |
| `goodmemDeleteSpace` | Permanently delete a space and all its memories |
| `goodmemCreateMemory` | Store text or files as memories |
| `goodmemListMemories` | List memories within a space (paginated, filterable) |
| `goodmemRetrieveMemories` | Semantic search with optional reranker / LLM post-processor |
| `goodmemGetMemory` | Fetch a specific memory by ID |
| `goodmemDeleteMemory` | Permanently delete a memory |

You can also call any tool directly without an agent:

```java
GoodMemTools tools = GoodMemTools.builder()
        .baseUrl("https://localhost:8080")
        .apiKey("your-api-key")
        .verifySsl(false)
        .build();

String embedders = tools.goodmemListEmbedders();
String space = tools.goodmemCreateSpace("my-space", "embedder-id", null, null, null);
String memory = tools.goodmemCreateMemory("space-id", "Important information.", null);
String results = tools.goodmemRetrieveMemories(
        "search query", "space-id", 5, true, true,
        null, null, null, null, null);
```

See [`GoodMemTools.java`](src/main/java/ai/pairsys/goodmem/langchain4j/GoodMemTools.java) for the full method signatures.

## Testing

Unit-style tests run by default. Integration tests (`*IT.java`) require a live GoodMem server and OpenAI API key, and are auto-skipped when the env vars below are unset:

```bash
export GOODMEM_BASE_URL=https://localhost:8080
export GOODMEM_API_KEY=your-key
export OPENAI_API_KEY=sk-...
./mvnw verify
```

## Project Structure

```
src/
├── main/java/ai/pairsys/goodmem/langchain4j/
│   ├── GoodMemClient.java     # HTTP client for the GoodMem REST API
│   ├── GoodMemException.java  # Wrapper exception for API errors
│   └── GoodMemTools.java      # LangChain4j @Tool surface
└── test/java/ai/pairsys/goodmem/langchain4j/
    ├── GoodMemConversationIT.java  # End-to-end agent conversation IT
    └── GoodMemToolsIT.java         # Tool-level integration tests
```

## License

[MIT](LICENSE) © PAIR Systems, Inc.
