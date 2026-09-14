package ai.pairsys.goodmem.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.pairsys.goodmem.client.models.GoodMemStatusCode;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.rag.content.ContentMetadata;
import dev.langchain4j.rag.query.Query;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class GoodMemContentRetrieverTest extends SdkTestSupport {
  @Test
  void joinsLateMetadataPreservesOrderAndOpaqueScoresAndDeduplicates() throws Exception {
    events(
        chunk(CHUNK, MEMORY, "First", -0.62),
        chunk(CHUNK_2, MEMORY, "Second", -0.52),
        chunk(CHUNK, MEMORY, "First", -0.62),
        definition(
            MEMORY,
            SPACE,
            Map.of("source", "https://example.com/doc", "title", "Guide", "edition", 3)));
    var contents = retriever().build().retrieve(Query.from("question"));
    assertEquals(
        List.of("First", "Second"), contents.stream().map(c -> c.textSegment().text()).toList());
    assertEquals(-0.62, contents.getFirst().metadata().get(ContentMetadata.SCORE));
    var metadata = contents.getFirst().textSegment().metadata();
    assertEquals("https://example.com/doc", metadata.getString("source"));
    assertEquals("Guide", metadata.getString("title"));
    assertEquals(3, metadata.getInteger("edition"));
    assertEquals(MEMORY, metadata.getString("memory_id"));
    assertEquals(CHUNK, metadata.getString("chunk_id"));
    assertEquals(SPACE, metadata.getString("space_id"));
    var sent = body(request());
    assertTrue(sent.get("fetchMemory").asBoolean());
    assertFalse(sent.get("fetchMemoryContent").asBoolean());
  }

  @Test
  void reranksWithoutLlmAndCombinesFixedAndDynamicFiltersAcrossSpaces() throws Exception {
    String fixed = "CAST(val('$.tenant') AS TEXT) = 'team'";
    String dynamic = "CAST(val('$.kind') AS TEXT) = 'guide'";
    events(
        chunk(CHUNK, MEMORY, "Relevant", 0.8),
        definition(MEMORY, SPACE, Map.of()),
        json(
            Map.of(
                "status",
                Map.of(
                    "code",
                    "FEATURE_DISABLED",
                    "details",
                    Map.of("feature", "summarization", "required_param", "llm_id")))));
    var contents =
        retriever()
            .spaceIds(List.of(SPACE, SPACE_2))
            .maxResults(2)
            .fetchK(12)
            .rerankerId(RERANKER)
            .filterExpression(fixed)
            .dynamicFilterExpression(query -> dynamic)
            .build()
            .retrieve(Query.from("question"));
    assertEquals(0.8, contents.getFirst().metadata().get(ContentMetadata.RERANKED_SCORE));
    var sent = body(request());
    assertEquals(12, sent.get("requestedSize").asInt());
    assertEquals(2, sent.get("spaceKeys").size());
    sent.get("spaceKeys")
        .forEach(
            key ->
                assertEquals("(" + fixed + ") AND (" + dynamic + ")", key.get("filter").asText()));
    var config = sent.at("/postProcessor/config");
    assertEquals(RERANKER, config.get("reranker_id").asText());
    assertEquals(2, config.get("max_results").asInt());
    assertFalse(config.has("llm_id"));
  }

  @Test
  void agentCannotOverrideConfiguredSpacesOrFilters() throws Exception {
    var tool = retriever().filterExpression("tenant-filter").build().asTool();
    var spec = tool.toolSpecification();
    assertEquals(List.of("query"), spec.parameters().required());
    assertEquals(java.util.Set.of("query"), spec.parameters().properties().keySet());
    events(
        chunk(CHUNK, MEMORY, "Found text", 0.5),
        definition(MEMORY, SPACE, Map.of("source", "docs")));
    var invocation =
        ToolExecutionRequest.builder()
            .id("one")
            .name("goodmemSearch")
            .arguments(
                json(Map.of("query", "find text", "spaceIds", List.of(SPACE_2), "filter", "TRUE")))
            .build();
    assertThrows(
        IllegalArgumentException.class, () -> tool.toolExecutor().execute(invocation, null));
    assertEquals(0, server.getRequestCount());
    var valid =
        ToolExecutionRequest.builder()
            .id("two")
            .name("goodmemSearch")
            .arguments(json(Map.of("query", "find text")))
            .build();
    String result = tool.toolExecutor().execute(valid, null);
    assertTrue(result.contains("Found text"));
    assertTrue(result.contains("docs"));
    var sent = body(request());
    assertEquals(SPACE, sent.at("/spaceKeys/0/spaceId").asText());
    assertEquals("tenant-filter", sent.at("/spaceKeys/0/filter").asText());
  }

  @Test
  void emptySearchIsOneRequestWithoutIndexingPolling() {
    events("{}");
    assertEquals(List.of(), retriever().build().retrieve(Query.from("absent")));
    assertEquals(1, server.getRequestCount());
  }

  @ParameterizedTest
  @ValueSource(
      strings = {
        "EMBEDDER_FAILED",
        "RERANKING_FAILED",
        "VECTOR_SEARCH_PARTIAL",
        "VECTOR_SEARCH_FAILED",
        "MEMORY_CONTENT_UNAVAILABLE",
        "SUMMARIZATION_FAILED",
        "NOT_FOUND"
      })
  void incompleteRetrievalIsAnErrorEvenWhenChunksExist(String code) {
    events(
        chunk(CHUNK, MEMORY, "Partial text", 0.5),
        definition(MEMORY, SPACE, Map.of()),
        status(code));
    var failure =
        assertThrows(
            GoodMemRetrievalException.class,
            () -> retriever().build().retrieve(Query.from("query")));
    assertEquals(GoodMemStatusCode.valueOf(code), failure.statuses().getFirst().code());
    assertEquals(1, server.getRequestCount());
  }

  @Test
  void sdkProtocolErrorsAreNotTurnedIntoSuccessfulResults() {
    events(
        chunk(CHUNK, MEMORY, "Partial", 0.5), "{invalid-json", definition(MEMORY, SPACE, Map.of()));
    assertThrows(RuntimeException.class, () -> retriever().build().retrieve(Query.from("query")));
    assertEquals(1, server.getRequestCount());
  }

  @Test
  void futureStatusDoesNotDiscardValidRetrievalResults() {
    events(
        chunk(CHUNK, MEMORY, "Before the unfamiliar status", 0.6),
        status("FUTURE_INFORMATIONAL_STATUS"),
        chunk(CHUNK_2, MEMORY, "After the unfamiliar status", 0.5),
        definition(MEMORY, SPACE, Map.of()));
    var results = retriever().build().retrieve(Query.from("query"));
    assertEquals(
        List.of("Before the unfamiliar status", "After the unfamiliar status"),
        results.stream().map(content -> content.textSegment().text()).toList());
    assertEquals(1, server.getRequestCount());
  }

  @Test
  void missingSourceMetadataAndUnexpectedSpaceFailExplicitly() {
    events(chunk(CHUNK, MEMORY, "Text", 0.5));
    assertThrows(GoodMemException.class, () -> retriever().build().retrieve(Query.from("query")));
    events(chunk(CHUNK, MEMORY, "Text", 0.5), definition(MEMORY, SPACE_2, Map.of()));
    assertThrows(GoodMemException.class, () -> retriever().build().retrieve(Query.from("query")));
  }

  @Test
  void representsJsonMetadataInLangchainTypesWithoutInventingSource() {
    events(
        chunk(CHUNK, MEMORY, "Text", 0.5),
        definition(
            MEMORY,
            SPACE,
            Map.of("enabled", true, "tags", List.of("a", "b"), "nested", Map.of("revision", 1))));
    var metadata =
        retriever().build().retrieve(Query.from("query")).getFirst().textSegment().metadata();
    assertEquals("true", metadata.getString("enabled"));
    assertEquals("[\"a\",\"b\"]", metadata.getString("tags"));
    assertEquals("{\"revision\":1}", metadata.getString("nested"));
    assertFalse(metadata.containsKey("source"));
  }

  @Test
  void preservesOriginalMetadataWhenChunkFieldsOrCanonicalIdsCollide() throws Exception {
    var chunkEvent = JSON.readTree(chunk(CHUNK, MEMORY, "Text", 0.5));
    ((com.fasterxml.jackson.databind.node.ObjectNode) chunkEvent.at("/retrievedItem/chunk/chunk"))
        .set(
            "metadata", JSON.valueToTree(Map.of("title", "Chunk title", "chunk_id", "user chunk")));
    events(
        chunkEvent.toString(),
        definition(
            MEMORY,
            SPACE,
            Map.of(
                "title",
                "Memory title",
                "memory_id",
                "user memory",
                "space_id",
                "user space",
                "goodmem_metadata",
                "User's own field",
                "source",
                "docs")));
    var metadata =
        retriever().build().retrieve(Query.from("query")).getFirst().textSegment().metadata();
    assertEquals("Chunk title", metadata.getString("title"));
    assertEquals(MEMORY, metadata.getString("memory_id"));
    assertEquals(CHUNK, metadata.getString("chunk_id"));
    assertEquals(SPACE, metadata.getString("space_id"));
    assertEquals("docs", metadata.getString("source"));
    var originals = JSON.readTree(metadata.getString("goodmem_metadata"));
    assertEquals("Memory title", originals.at("/memory/title").asText());
    assertEquals("Chunk title", originals.at("/chunk/title").asText());
    assertEquals("user memory", originals.at("/memory/memory_id").asText());
    assertEquals("user chunk", originals.at("/chunk/chunk_id").asText());
    assertEquals("user space", originals.at("/memory/space_id").asText());
    assertEquals("User's own field", originals.at("/memory/goodmem_metadata").asText());
  }

  @Test
  void validatesConfigurationBeforeAnyHttpRequest() {
    assertThrows(IllegalArgumentException.class, () -> retriever().spaceIds(List.of()).build());
    assertThrows(IllegalArgumentException.class, () -> retriever().maxResults(0).build());
    assertThrows(IllegalArgumentException.class, () -> retriever().maxResults(5).fetchK(4).build());
    assertEquals(0, server.getRequestCount());
  }
}
