package ai.pairsys.goodmem.langchain4j;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.pairsys.goodmem.client.errors.AuthenticationException;
import ai.pairsys.goodmem.client.models.GoodMemStatusCode;
import ai.pairsys.goodmem.client.models.MemoriesListStatusFilter;
import ai.pairsys.goodmem.client.models.MemoryProcessingStatus;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecifications;
import dev.langchain4j.service.tool.DefaultToolExecutor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import okhttp3.mockwebserver.MockResponse;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class GoodMemToolsTest extends SdkTestSupport {
  GoodMemTools tools() {
    return new GoodMemTools(client);
  }

  @Test
  void createSpaceUsesSdkDefaultsWithoutListingOrReusingByName() throws Exception {
    reply(space(SPACE));
    assertEquals(
        SPACE,
        tools()
            .goodmemCreateSpace("same name", EMBEDDER, Map.of("team", "docs"))
            .spaceId()
            .toString());
    var sent = request();
    assertEquals("POST", sent.getMethod());
    assertEquals("/v1/spaces", sent.getPath());
    var payload = body(sent);
    assertEquals(EMBEDDER, payload.at("/spaceEmbedders/0/embedderId").asText());
    assertTrue(payload.has("defaultChunkingConfig"));
    assertEquals(1, server.getRequestCount());
  }

  @Test
  void followsSpacePaginationAndKeepsNameFilter() throws Exception {
    reply("{\"spaces\":[" + space(SPACE) + "],\"nextToken\":\"page-two\"}");
    reply("{\"spaces\":[" + space(SPACE_2) + "]}");
    assertEquals(2, tools().goodmemListSpaces("project", null).size());
    assertEquals("project", request().getRequestUrl().queryParameter("nameFilter"));
    var second = request();
    assertEquals("project", second.getRequestUrl().queryParameter("nameFilter"));
    assertEquals("page-two", second.getRequestUrl().queryParameter("nextToken"));
  }

  @Test
  void respectsListingLimitWithoutFetchingAnotherPage() {
    reply("{\"spaces\":[" + space(SPACE) + "],\"nextToken\":\"page-two\"}");
    assertEquals(1, tools().goodmemListSpaces(null, 1).size());
    assertEquals(1, server.getRequestCount());
  }

  @Test
  void memoryPaginationPreservesMetadataAndProcessingFilters() throws Exception {
    reply(
        json(
            Map.of(
                "memories",
                List.of(memory(MEMORY, SPACE, "COMPLETED", Map.of())),
                "nextToken",
                "two")));
    reply(json(Map.of("memories", List.of(memory(MEMORY_2, SPACE, "COMPLETED", Map.of())))));
    assertEquals(
        2,
        tools()
            .goodmemListMemories(SPACE, null, MemoriesListStatusFilter.COMPLETED, "example-filter")
            .size());
    for (int i = 0; i < 2; i++) {
      var sent = request();
      assertEquals("example-filter", sent.getRequestUrl().queryParameter("filter"));
      assertEquals("COMPLETED", sent.getRequestUrl().queryParameter("statusFilter"));
    }
  }

  @Test
  void rawRetrievalPreservesFailuresAlongsideChunksAndIsNeverPolled() throws Exception {
    events(chunk(CHUNK, MEMORY, "Fallback", 0.4), status("RERANKING_FAILED"));
    var result =
        tools()
            .goodmemRetrieveMemories("question", List.of(SPACE), 3, 12, "filter", RERANKER, null);
    assertEquals("Fallback", result.chunks().getFirst().text());
    assertTrue(result.partial());
    assertEquals(GoodMemStatusCode.RERANKING_FAILED, result.statuses().getFirst().code());
    var sent = body(request());
    assertEquals(12, sent.get("requestedSize").asInt());
    assertEquals("filter", sent.at("/spaceKeys/0/filter").asText());
    assertFalse(sent.at("/postProcessor/config").has("llm_id"));
    events(status("EMBEDDER_FAILED"));
    assertEquals(
        GoodMemStatusCode.EMBEDDER_FAILED,
        tools()
            .goodmemRetrieveMemories("question", List.of(SPACE), null, null, null, null, null)
            .statuses()
            .getFirst()
            .code());
    assertEquals(2, server.getRequestCount());
  }

  @Test
  void rawRetrievalSupportsOptionalLlmRepliesAndFrameworkSerialization() throws Exception {
    events(
        chunk(CHUNK, MEMORY, "Supporting evidence", 0.6),
        json(Map.of("abstractReply", Map.of("text", "The answer from the configured LLM."))));
    var invocation =
        ToolExecutionRequest.builder()
            .id("summary")
            .name("goodmemRetrieveMemories")
            .arguments(
                json(Map.of("query", "question", "spaceIds", List.of(SPACE), "llmId", RERANKER)))
            .build();
    var result =
        JSON.readTree(new DefaultToolExecutor(tools(), invocation).execute(invocation, null));
    assertEquals("The answer from the configured LLM.", result.get("abstractReply").asText());
    assertEquals("Supporting evidence", result.at("/chunks/0/text").asText());
    assertTrue(result.get("partial").asBoolean(), "Missing memory metadata must be visible");
    assertFalse(result.has("memoryDefinition"));
    var sent = body(request());
    assertEquals(RERANKER, sent.at("/postProcessor/config/llm_id").asText());
    assertFalse(sent.at("/postProcessor/config").has("reranker_id"));
    assertEquals(1, server.getRequestCount());
  }

  @Test
  void futureStatusIsPreservedWithoutLosingOtherEvents() {
    events(
        chunk(CHUNK, MEMORY, "Before the unfamiliar status", 0.6),
        status("FUTURE_STATUS"),
        chunk(CHUNK_2, MEMORY, "After the unfamiliar status", 0.5),
        definition(MEMORY, SPACE, Map.of("source", "docs")));
    var results =
        tools().goodmemRetrieveMemories("query", List.of(SPACE), null, null, null, null, null);
    assertEquals(GoodMemStatusCode.UNKNOWN, results.statuses().getFirst().code());
    assertEquals(
        List.of("Before the unfamiliar status", "After the unfamiliar status"),
        results.chunks().stream().map(GoodMemTools.RetrievedChunk::text).toList());
    assertTrue(results.partial());
  }

  @Test
  void fetchKIsSentWithoutPostProcessingAndReturnedChunksStillRespectMaxResults() throws Exception {
    events(
        chunk(CHUNK, MEMORY, "First", 0.7),
        chunk(CHUNK_2, MEMORY, "Second", 0.6),
        definition(MEMORY, SPACE, Map.of("source", "docs")));
    var result = tools().goodmemRetrieveMemories("query", List.of(SPACE), 1, 12, null, null, null);
    assertEquals(1, result.chunks().size());
    assertFalse(result.partial());
    var payload = body(request());
    assertEquals(12, payload.get("requestedSize").asInt());
    assertFalse(payload.has("postProcessor"));
  }

  @Test
  void sdkAuthenticationFailurePropagates() {
    server.enqueue(
        new MockResponse().setResponseCode(401).setBody("{\"message\":\"invalid key\"}"));
    assertThrows(AuthenticationException.class, () -> tools().goodmemGetSpace(SPACE));
  }

  @Test
  void toolSchemasRemoveObsoleteArgumentsAndUseNativeJsonMetadata() {
    var specs = ToolSpecifications.toolSpecificationsFrom(tools());
    assertEquals(11, specs.size());
    var update =
        specs.stream().filter(s -> s.name().equals("goodmemUpdateSpace")).findFirst().orElseThrow();
    assertFalse(update.parameters().properties().containsKey("publicRead"));
    var create =
        specs.stream().filter(s -> s.name().equals("goodmemCreateSpace")).findFirst().orElseThrow();
    assertEquals(
        java.util.Set.of("name", "embedderId", "labels"),
        create.parameters().properties().keySet());
    var retrieve =
        specs.stream()
            .filter(s -> s.name().equals("goodmemRetrieveMemories"))
            .findFirst()
            .orElseThrow();
    assertFalse(retrieve.parameters().properties().containsKey("waitForIndexing"));
  }

  @Test
  void frameworkSerializesSdkIdsAsStringsAndReportsCompletedWrites() throws Exception {
    reply(json(memory(MEMORY, SPACE, "PENDING", Map.of())));
    reply(json(memory(MEMORY, SPACE, "COMPLETED", Map.of("source", "agent"))));
    var invocation =
        ToolExecutionRequest.builder()
            .id("one")
            .name("goodmemCreateMemory")
            .arguments(
                json(
                    Map.of(
                        "spaceId",
                        SPACE,
                        "textContent",
                        "remember this",
                        "metadata",
                        Map.of("source", "agent"))))
            .build();
    var result =
        JSON.readTree(new DefaultToolExecutor(tools(), invocation).execute(invocation, null));
    assertEquals(MEMORY, result.get("memoryId").asText());
    assertEquals("COMPLETED", result.get("processingStatus").asText());
    assertFalse(result.has("success"));
    assertEquals("agent", body(request()).at("/metadata/source").asText());
    assertEquals("/v1/memories/" + MEMORY, request().getPath());
  }

  @Test
  void uploadsBinaryFilesWithMetadataThroughSdkEncoding(@TempDir Path directory) throws Exception {
    byte[] bytes = new byte[] {0, 1, -1, -2};
    Path path = directory.resolve("sample.pdf");
    Files.write(path, bytes);
    reply(json(memory(MEMORY, SPACE, "PENDING", Map.of())));
    var result =
        new GoodMemTools(client, java.time.Duration.ofSeconds(60), directory)
            .goodmemCreateMemory(SPACE, null, path.toString(), Map.of("source", "upload"), false);
    assertEquals(MemoryProcessingStatus.PENDING, result.processingStatus());
    var sent = body(request());
    assertArrayEquals(bytes, Base64.getDecoder().decode(sent.get("originalContentB64").asText()));
    assertEquals("upload", sent.at("/metadata/source").asText());
    assertEquals(1, server.getRequestCount());
  }

  @Test
  void fileUploadsAreDisabledByDefaultAndCannotEscapeTheAllowedDirectory(@TempDir Path directory)
      throws Exception {
    Path allowed = Files.createDirectory(directory.resolve("uploads"));
    Path outside =
        Files.writeString(directory.resolve("private-config.txt"), "private test content");
    assertThrows(
        IllegalArgumentException.class,
        () -> tools().goodmemCreateMemory(SPACE, null, outside.toString(), null, false));
    var enabled = new GoodMemTools(client, java.time.Duration.ofSeconds(60), allowed);
    assertThrows(
        IllegalArgumentException.class,
        () -> enabled.goodmemCreateMemory(SPACE, null, outside.toString(), null, false));
    assertThrows(
        IllegalArgumentException.class,
        () -> enabled.goodmemCreateMemory(SPACE, null, "../private-config.txt", null, false));
    Files.createSymbolicLink(allowed.resolve("escape.txt"), outside);
    assertThrows(
        IllegalArgumentException.class,
        () -> enabled.goodmemCreateMemory(SPACE, null, "escape.txt", null, false));
    assertEquals(0, server.getRequestCount());
  }

  @Test
  void getBinaryContentGivesANoticeInsteadOfBase64() throws Exception {
    byte[] bytes = new byte[] {0, -1, 42};
    reply(
        json(
            Map.of(
                "memoryId",
                MEMORY,
                "spaceId",
                SPACE,
                "originalContent",
                Base64.getEncoder().encodeToString(bytes))));
    var result = tools().goodmemGetMemory(MEMORY, true);
    assertNull(result.text());
    assertTrue(result.contentNotice().contains("Binary content omitted"));
    var sent = request();
    assertEquals("/v1/memories/" + MEMORY, sent.getRequestUrl().encodedPath());
    assertEquals("true", sent.getRequestUrl().queryParameter("includeContent"));
    assertEquals(1, server.getRequestCount());
  }

  @Test
  void getTextIsReadableByDefaultThroughTheActualToolExecutor() throws Exception {
    reply(
        json(
            Map.of(
                "memoryId",
                MEMORY,
                "spaceId",
                SPACE,
                "contentType",
                "text/plain; charset=UTF-8",
                "originalContent",
                Base64.getEncoder()
                    .encodeToString(
                        "A café note".getBytes(java.nio.charset.StandardCharsets.UTF_8)))));
    var invocation =
        ToolExecutionRequest.builder()
            .id("read")
            .name("goodmemGetMemory")
            .arguments(json(Map.of("memoryId", MEMORY)))
            .build();
    var result =
        JSON.readTree(new DefaultToolExecutor(tools(), invocation).execute(invocation, null));
    assertEquals("A café note", result.get("text").asText());
    assertFalse(result.has("originalContent"));
    assertEquals("true", request().getRequestUrl().queryParameter("includeContent"));
  }

  @Test
  void compactRetrievalJoinsSourceAndKeepsSuccessfulEmptyResults() {
    events(
        chunk(CHUNK, MEMORY, "Evidence", 0.7),
        definition(MEMORY, SPACE, Map.of("source", "handbook")));
    var result =
        tools().goodmemRetrieveMemories("query", List.of(SPACE), 3, null, null, null, null);
    assertFalse(result.partial());
    assertEquals("handbook", result.chunks().getFirst().source());
    assertEquals(0.7, result.chunks().getFirst().score());
    events("{}");
    var empty =
        tools().goodmemRetrieveMemories("empty", List.of(SPACE), null, null, null, null, null);
    assertTrue(empty.chunks().isEmpty());
    assertFalse(empty.partial());
  }

  @Test
  void updateAndDeleteUseSdkOperations() throws Exception {
    reply(space(SPACE));
    tools().goodmemUpdateSpace(SPACE, "new name", null, Map.of("team", "docs"));
    var sent = request();
    assertEquals("PUT", sent.getMethod());
    assertFalse(body(sent).has("publicRead"));
    server.enqueue(new MockResponse().setResponseCode(204));
    tools().goodmemDeleteMemory(MEMORY);
    assertEquals("/v1/memories/" + MEMORY, request().getPath());
    server.enqueue(new MockResponse().setResponseCode(204));
    tools().goodmemDeleteSpace(SPACE);
    assertEquals("/v1/spaces/" + SPACE, request().getPath());
    reply("{\"embedders\":[]}");
    assertEquals(List.of(), tools().goodmemListEmbedders());
  }
}
