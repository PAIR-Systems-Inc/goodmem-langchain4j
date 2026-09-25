package ai.pairsys.goodmem.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.ContentMetadata;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Result;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import org.junit.jupiter.api.Test;

/**
 * Q4a/Q4b: a server diagnostic never discards hits and never turns an empty search into an error.
 */
class GoodMemPartialRetrievalTest extends SdkTestSupport {
  interface Assistant {
    Result<String> chat(String question);
  }

  static String chunkId(int index) {
    return "33333333-3333-4333-8333-3333333330%02d".formatted(index);
  }

  static String status(String code, String message) {
    return json(Map.of("status", Map.of("code", code, "message", message)));
  }

  /** Vector score of the i-th fallback hit: negative inner product, so the best is the lowest. */
  static double vectorScore(int index) {
    return -0.7168 + index * 0.03;
  }

  /**
   * The stream GoodMem sent for a nonexistent reranker (captured from a live server): diagnostics,
   * the informational FEATURE_DISABLED for summarization, a result set produced by the "retrieve"
   * stage, and ten vector hits in ascending (best-first) negative inner-product order.
   */
  void rerankerFallbackStream() {
    List<String> lines = new ArrayList<>();
    lines.add(status("NOT_FOUND", "Reranker not found"));
    lines.add(
        json(
            Map.of(
                "status",
                Map.of(
                    "code",
                    "FEATURE_DISABLED",
                    "message",
                    "Abstract reply generation disabled: no LLM configured.",
                    "details",
                    Map.of("required_param", "llm_id", "feature", "summarization")))));
    lines.add(status("RERANKING_FAILED", "Reranking skipped; returning vector results"));
    lines.add(GoodMemIncompleteStreamTest.boundary("BEGIN", "retrieve"));
    lines.add(definition(MEMORY, SPACE, Map.of("source", "handbook")));
    for (int i = 0; i < 10; i++) {
      lines.add(chunk(chunkId(i), MEMORY, "Vector hit " + i, vectorScore(i)));
    }
    lines.add(GoodMemIncompleteStreamTest.boundary("END", "retrieve"));
    events(lines.toArray(String[]::new));
  }

  /** A normal three-hit stream with a RERANKING_FAILED status injected between hits. */
  void injectedStatusStream() {
    events(
        chunk(chunkId(0), MEMORY, "First", 0.8),
        status("RERANKING_FAILED", "Injected by proxy"),
        chunk(chunkId(1), MEMORY, "Second", 0.7),
        chunk(chunkId(2), MEMORY, "Third", 0.6),
        definition(MEMORY, SPACE, Map.of()));
  }

  static List<String> codes(Content content) throws Exception {
    var statuses = JSON.readTree(content.textSegment().metadata().getString("goodmem_statuses"));
    List<String> codes = new ArrayList<>();
    statuses.forEach(status -> codes.add(status.get("code").asText()));
    return codes;
  }

  static <T> Captured<T> captureStderr(Supplier<T> action) {
    PrintStream original = System.err;
    var buffer = new ByteArrayOutputStream();
    System.setErr(new PrintStream(buffer, true, StandardCharsets.UTF_8));
    try {
      T value = action.get();
      return new Captured<>(value, buffer.toString(StandardCharsets.UTF_8));
    } finally {
      System.setErr(original);
    }
  }

  record Captured<T>(T value, String stderr) {
    long warnings() {
      return stderr
          .lines()
          .filter(line -> line.contains("WARN") && line.contains("RetrievalResults"))
          .count();
    }
  }

  @Test
  void rerankerFallbackKeepsAllTenVectorHitsAndMarksThemPartial() throws Exception {
    rerankerFallbackStream();
    var captured =
        captureStderr(
            () ->
                retriever()
                    .rerankerId(RERANKER)
                    .maxResults(10)
                    .build()
                    .retrieve(Query.from("question")));
    var contents = captured.value();
    assertEquals(10, contents.size());
    for (int i = 0; i < 10; i++) {
      var content = contents.get(i);
      assertEquals("Vector hit " + i, content.textSegment().text());
      var metadata = content.textSegment().metadata();
      assertEquals("true", metadata.getString("goodmem_partial"));
      assertEquals(List.of("NOT_FOUND", "RERANKING_FAILED"), codes(content));
      // The reranker never ran: these are vector scores and must not be labelled as reranked.
      assertEquals("vector", metadata.getString("goodmem_score_type"));
      assertEquals(vectorScore(i), content.metadata().get(ContentMetadata.SCORE));
      assertNull(content.metadata().get(ContentMetadata.RERANKED_SCORE));
      assertEquals("handbook", metadata.getString("source"));
    }
    assertEquals(1, captured.warnings(), captured.stderr());
    assertTrue(captured.stderr().contains("RERANKING_FAILED"), captured.stderr());
    assertEquals(1, server.getRequestCount());
  }

  @Test
  void injectedRerankingFailedInANormalStreamKeepsEveryHit() throws Exception {
    injectedStatusStream();
    var contents = retriever().build().retrieve(Query.from("question"));
    assertEquals(
        List.of("First", "Second", "Third"),
        contents.stream().map(c -> c.textSegment().text()).toList());
    for (var content : contents) {
      assertEquals("true", content.textSegment().metadata().getString("goodmem_partial"));
      assertEquals(List.of("RERANKING_FAILED"), codes(content));
    }
  }

  @Test
  void problemWithoutHitsIsAnEmptyListNotAnException() {
    events(status("EMBEDDER_FAILED", "Embedder unavailable"));
    var captured = captureStderr(() -> retriever().build().retrieve(Query.from("question")));
    assertEquals(List.of(), captured.value());
    assertEquals(1, captured.warnings(), captured.stderr());
    assertTrue(captured.stderr().contains("EMBEDDER_FAILED"), captured.stderr());
  }

  @Test
  void completeRetrievalCarriesNoPartialMarkerAndNoWarning() {
    events(chunk(CHUNK, MEMORY, "Complete", 0.8), definition(MEMORY, SPACE, Map.of()));
    var captured =
        captureStderr(
            () -> retriever().rerankerId(RERANKER).build().retrieve(Query.from("question")));
    var content = captured.value().getFirst();
    var metadata = content.textSegment().metadata();
    assertFalse(metadata.containsKey("goodmem_partial"));
    assertFalse(metadata.containsKey("goodmem_statuses"));
    assertEquals("reranker", metadata.getString("goodmem_score_type"));
    assertEquals(0.8, content.metadata().get(ContentMetadata.RERANKED_SCORE));
    assertEquals(0, captured.warnings(), captured.stderr());
  }

  @Test
  void userMetadataCannotForgeOrHideThePartialMarker() throws Exception {
    events(
        chunk(CHUNK, MEMORY, "Complete", 0.8),
        definition(
            MEMORY,
            SPACE,
            Map.of("goodmem_partial", "true", "goodmem_score_type", "reranker", "team", "x")));
    var metadata =
        retriever().build().retrieve(Query.from("question")).getFirst().textSegment().metadata();
    assertFalse(metadata.containsKey("goodmem_partial"));
    assertEquals("vector", metadata.getString("goodmem_score_type"));
    var originals = JSON.readTree(metadata.getString("goodmem_metadata"));
    assertEquals("true", originals.at("/memory/goodmem_partial").asText());
  }

  @Test
  void aiServiceAnswersFromFallbackHitsInsteadOfFailingTheChat() {
    rerankerFallbackStream();
    ChatModel model =
        new ChatModel() {
          @Override
          public ChatResponse doChat(ChatRequest request) {
            assertTrue(request.messages().toString().contains("Vector hit 0"));
            return ChatResponse.builder().aiMessage(AiMessage.from("Answered")).build();
          }
        };
    var assistant =
        AiServices.builder(Assistant.class)
            .chatModel(model)
            .contentRetriever(retriever().rerankerId(RERANKER).maxResults(10).build())
            .build();
    var answer = assistant.chat("question");
    assertEquals("Answered", answer.content());
    assertEquals(10, answer.sources().size());
    assertEquals(
        "true", answer.sources().getFirst().textSegment().metadata().getString("goodmem_partial"));
  }

  @Test
  void searchToolHandsTheModelTheHitsAndAPartialNote() throws Exception {
    rerankerFallbackStream();
    var tool = retriever().rerankerId(RERANKER).maxResults(10).build().asTool();
    String result =
        tool.toolExecutor()
            .execute(
                ToolExecutionRequest.builder()
                    .id("one")
                    .name("goodmemSearch")
                    .arguments(json(Map.of("query", "question")))
                    .build(),
                null);
    var output = JSON.readTree(result);
    assertTrue(output.isObject(), result);
    assertTrue(output.get("partial").asBoolean(), result);
    assertEquals(10, output.get("results").size(), result);
    assertEquals("Vector hit 0", output.at("/results/0/text").asText());
    assertEquals("handbook", output.at("/results/0/metadata/source").asText());
    assertEquals("NOT_FOUND", output.at("/statuses/0/code").asText());
    assertEquals("RERANKING_FAILED", output.at("/statuses/1/code").asText());
    assertTrue(output.get("note").asText().contains("incomplete"), result);
    assertTrue(output.get("note").asText().contains("vector"), result);
  }

  @Test
  void searchToolReportsAnEmptyPartialResultWhenNoHitsSurvive() throws Exception {
    events(status("VECTOR_SEARCH_FAILED", "Index offline"));
    var tool = retriever().build().asTool();
    String result =
        tool.toolExecutor()
            .execute(
                ToolExecutionRequest.builder()
                    .id("one")
                    .name("goodmemSearch")
                    .arguments(json(Map.of("query", "question")))
                    .build(),
                null);
    var output = JSON.readTree(result);
    assertTrue(output.get("partial").asBoolean(), result);
    assertEquals(0, output.get("results").size(), result);
    assertEquals("VECTOR_SEARCH_FAILED", output.at("/statuses/0/code").asText());
  }

  @Test
  void searchToolOutputIsUnchangedForCompleteResults() throws Exception {
    events(chunk(CHUNK, MEMORY, "Complete", 0.8), definition(MEMORY, SPACE, Map.of()));
    String result =
        retriever()
            .build()
            .asTool()
            .toolExecutor()
            .execute(
                ToolExecutionRequest.builder()
                    .id("one")
                    .name("goodmemSearch")
                    .arguments(json(Map.of("query", "question")))
                    .build(),
                null);
    var output = JSON.readTree(result);
    assertTrue(output.isArray(), result);
    assertEquals(1, output.size(), result);
    assertEquals("Complete", output.at("/0/text").asText());
    // The 0.2.0 keys, plus goodmem_score_type (new in 0.2.1); no partial markers.
    List<String> keys = new ArrayList<>();
    output.at("/0/metadata").fieldNames().forEachRemaining(keys::add);
    assertEquals(
        java.util.Set.of("memory_id", "chunk_id", "space_id", "goodmem_score_type"),
        java.util.Set.copyOf(keys),
        result);
    assertEquals(
        java.util.Set.of("text", "metadata"), java.util.Set.copyOf(fieldNames(output.get(0))));
  }

  static List<String> fieldNames(com.fasterxml.jackson.databind.JsonNode node) {
    List<String> names = new ArrayList<>();
    node.fieldNames().forEachRemaining(names::add);
    return names;
  }

  @Test
  void searchToolNoteDoesNotBlameRerankingWhenNoRerankerIsConfigured() throws Exception {
    injectedStatusStream();
    String result =
        retriever()
            .build()
            .asTool()
            .toolExecutor()
            .execute(
                ToolExecutionRequest.builder()
                    .id("one")
                    .name("goodmemSearch")
                    .arguments(json(Map.of("query", "question")))
                    .build(),
                null);
    var output = JSON.readTree(result);
    assertEquals(3, output.get("results").size(), result);
    assertFalse(output.get("note").asText().contains("vector-search order"), result);
    assertFalse(output.at("/results/0/metadata").has("goodmem_statuses"), result);
  }

  @Test
  void retrieverAndAdministrativeToolAgreeOnTheFallbackStream() {
    rerankerFallbackStream();
    var admin =
        new GoodMemTools(client)
            .goodmemRetrieveMemories("question", List.of(SPACE), 10, null, null, RERANKER, null);
    rerankerFallbackStream();
    var contents =
        retriever().rerankerId(RERANKER).maxResults(10).build().retrieve(Query.from("question"));
    assertTrue(admin.partial());
    assertEquals(admin.chunks().size(), contents.size());
    assertEquals(
        admin.chunks().stream().map(GoodMemTools.RetrievedChunk::text).toList(),
        contents.stream().map(c -> c.textSegment().text()).toList());
  }
}
