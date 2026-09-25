package ai.pairsys.goodmem.langchain4j;

import static ai.pairsys.goodmem.langchain4j.GoodMemPartialRetrievalTest.captureStderr;
import static ai.pairsys.goodmem.langchain4j.GoodMemPartialRetrievalTest.codes;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Q4a for streams whose items are themselves incomplete, Q1 for {@code FEATURE_DISABLED}, and how
 * scores are labelled when the stream says which stage produced them.
 */
class GoodMemIncompleteStreamTest extends SdkTestSupport {
  static String status(String code, String message, Map<String, ?> details) {
    Map<String, Object> status = new LinkedHashMap<>();
    status.put("code", code);
    status.put("message", message);
    if (details != null) {
      status.put("details", details);
    }
    return json(Map.of("status", status));
  }

  /** A retrieved item whose chunk carries identifiers but no text, as the server can send. */
  static String chunkWithoutText(String id, String memory, double score) {
    return json(
        Map.of(
            "retrievedItem",
            Map.of(
                "chunk",
                Map.of(
                    "chunk", Map.of("chunkId", id, "memoryId", memory), "relevanceScore", score))));
  }

  static String boundary(String kind, String stageName) {
    return json(
        Map.of(
            "resultSetBoundary",
            Map.of(
                "resultSetId",
                "66666666-6666-4666-8666-666666666666",
                "kind",
                kind,
                "stageName",
                stageName)));
  }

  static List<String> texts(List<Content> contents) {
    return contents.stream().map(c -> c.textSegment().text()).toList();
  }

  String search(GoodMemContentRetriever retriever) {
    return retriever
        .asTool()
        .toolExecutor()
        .execute(
            ToolExecutionRequest.builder()
                .id("one")
                .name("goodmemSearch")
                .arguments(json(Map.of("query", "question")))
                .build(),
            null);
  }

  /** MEMORY_CONTENT_UNAVAILABLE plus one good chunk and one chunk the server sent without text. */
  void contentUnavailableStream() {
    events(
        status("MEMORY_CONTENT_UNAVAILABLE", "Chunk text could not be loaded", null),
        boundary("BEGIN", "retrieve"),
        definition(MEMORY, SPACE, Map.of("source", "handbook")),
        chunk(CHUNK, MEMORY, "Good", -0.61),
        chunkWithoutText(CHUNK_2, MEMORY, -0.39),
        boundary("END", "retrieve"));
  }

  /** MEMORY_LOAD_FAILED: the second hit's memory has no definition in the stream. */
  void memoryLoadFailedStream() {
    events(
        status("MEMORY_LOAD_FAILED", "Memory 2 could not be loaded", null),
        boundary("BEGIN", "retrieve"),
        definition(MEMORY, SPACE, Map.of("source", "handbook")),
        chunk(CHUNK, MEMORY, "A", -0.61),
        chunk(CHUNK_2, MEMORY_2, "B", -0.39),
        boundary("END", "retrieve"));
  }

  @Test
  void chunkWithoutTextIsSkippedAndTheGoodHitIsKept() throws Exception {
    contentUnavailableStream();
    var captured = captureStderr(() -> retriever().build().retrieve(Query.from("question")));
    var contents = captured.value();
    assertEquals(List.of("Good"), texts(contents));
    var metadata = contents.getFirst().textSegment().metadata();
    assertEquals("true", metadata.getString("goodmem_partial"));
    assertEquals(List.of("MEMORY_CONTENT_UNAVAILABLE"), codes(contents.getFirst()));
    assertEquals("handbook", metadata.getString("source"));
    assertEquals(1, captured.warnings(), captured.stderr());
    assertTrue(captured.stderr().contains("without text"), captured.stderr());
  }

  @Test
  void hitWhoseMemoryHasNoDefinitionIsKeptWithTheOtherHits() throws Exception {
    memoryLoadFailedStream();
    var captured = captureStderr(() -> retriever().build().retrieve(Query.from("question")));
    var contents = captured.value();
    assertEquals(List.of("A", "B"), texts(contents));
    for (var content : contents) {
      assertEquals("true", content.textSegment().metadata().getString("goodmem_partial"));
      assertEquals(List.of("MEMORY_LOAD_FAILED"), codes(content));
    }
    var first = contents.get(0).textSegment().metadata();
    assertEquals(SPACE, first.getString("space_id"));
    assertEquals("handbook", first.getString("source"));
    var second = contents.get(1).textSegment().metadata();
    assertEquals(MEMORY_2, second.getString("memory_id"));
    assertEquals(CHUNK_2, second.getString("chunk_id"));
    // Unknown without the definition; never guessed.
    assertFalse(second.containsKey("space_id"));
    assertEquals(1, captured.warnings(), captured.stderr());
    assertTrue(captured.stderr().contains("without memory metadata"), captured.stderr());
  }

  @Test
  void retrieverAndAdministrativeToolAgreeOnIncompleteItems() {
    contentUnavailableStream();
    var admin =
        new GoodMemTools(client)
            .goodmemRetrieveMemories("question", List.of(SPACE), 5, null, null, null, null);
    contentUnavailableStream();
    var contents = retriever().build().retrieve(Query.from("question"));
    assertTrue(admin.partial());
    assertEquals(
        admin.chunks().stream().map(GoodMemTools.RetrievedChunk::text).toList(), texts(contents));

    memoryLoadFailedStream();
    admin =
        new GoodMemTools(client)
            .goodmemRetrieveMemories("question", List.of(SPACE), 5, null, null, null, null);
    memoryLoadFailedStream();
    contents = retriever().build().retrieve(Query.from("question"));
    assertTrue(admin.partial());
    assertEquals(
        admin.chunks().stream().map(GoodMemTools.RetrievedChunk::text).toList(), texts(contents));
  }

  @Test
  void aiServiceAnswersWhenAMemoryCouldNotBeLoaded() {
    memoryLoadFailedStream();
    ChatModel model =
        new ChatModel() {
          @Override
          public ChatResponse doChat(ChatRequest request) {
            return ChatResponse.builder().aiMessage(AiMessage.from("Answered")).build();
          }
        };
    var assistant =
        AiServices.builder(GoodMemPartialRetrievalTest.Assistant.class)
            .chatModel(model)
            .contentRetriever(retriever().build())
            .build();
    var answer = assistant.chat("question");
    assertEquals("Answered", answer.content());
    assertEquals(2, answer.sources().size());
  }

  @Test
  void incompleteItemsWithoutAnyStatusStillMarkTheResultPartial() throws Exception {
    events(
        definition(MEMORY, SPACE, Map.of()),
        chunkWithoutText(CHUNK_2, MEMORY, -0.5),
        chunk(CHUNK, MEMORY, "Good", -0.4));
    var captured = captureStderr(() -> retriever().build().retrieve(Query.from("question")));
    var metadata = captured.value().getFirst().textSegment().metadata();
    assertEquals(List.of("Good"), texts(captured.value()));
    assertEquals("true", metadata.getString("goodmem_partial"));
    assertEquals("[]", metadata.getString("goodmem_statuses"));
    assertEquals(1, captured.warnings(), captured.stderr());

    events(
        definition(MEMORY, SPACE, Map.of()),
        chunkWithoutText(CHUNK_2, MEMORY, -0.5),
        chunk(CHUNK, MEMORY, "Good", -0.4));
    var output = JSON.readTree(search(retriever().build()));
    assertTrue(output.get("partial").asBoolean(), output.toString());
    assertEquals(1, output.get("results").size(), output.toString());
    assertEquals(0, output.get("statuses").size(), output.toString());
    assertTrue(output.get("note").asText().contains("without text"), output.toString());
  }

  @Test
  void searchToolHandsTheModelTheKeptHitsForIncompleteItems() throws Exception {
    memoryLoadFailedStream();
    var output = JSON.readTree(search(retriever().build()));
    assertTrue(output.get("partial").asBoolean(), output.toString());
    assertEquals(2, output.get("results").size(), output.toString());
    assertEquals("MEMORY_LOAD_FAILED", output.at("/statuses/0/code").asText());
  }

  @Test
  void onlyIncompleteItemsIsAnEmptyListNotAnException() {
    events(
        status("MEMORY_CONTENT_UNAVAILABLE", "Nothing could be loaded", null),
        chunkWithoutText(CHUNK, MEMORY, -0.5));
    var captured = captureStderr(() -> retriever().build().retrieve(Query.from("question")));
    assertEquals(List.of(), captured.value());
    assertEquals(1, captured.warnings(), captured.stderr());
  }

  @Test
  void failOnIncompleteRetrievalStillRejectsIncompleteItems() {
    events(definition(MEMORY, SPACE, Map.of()), chunkWithoutText(CHUNK, MEMORY, -0.5));
    var textless =
        assertThrows(
            GoodMemException.class,
            () ->
                retriever()
                    .failOnIncompleteRetrieval(true)
                    .build()
                    .retrieve(Query.from("question")));
    assertTrue(textless.getMessage().contains("without text"), textless.getMessage());
    events(chunk(CHUNK, MEMORY, "Text", -0.5));
    var undefined =
        assertThrows(
            GoodMemException.class,
            () ->
                retriever()
                    .failOnIncompleteRetrieval(true)
                    .build()
                    .retrieve(Query.from("question")));
    assertTrue(undefined.getMessage().contains("omitted metadata"), undefined.getMessage());
  }

  @Test
  void aMemoryOutsideTheConfiguredSpacesIsStillRejected() {
    events(chunk(CHUNK, MEMORY, "Text", -0.5), definition(MEMORY, SPACE_2, Map.of()));
    assertThrows(GoodMemException.class, () -> retriever().build().retrieve(Query.from("query")));
  }

  static Stream<Arguments> featureDisabledDetails() {
    return Stream.of(
        Arguments.of("no details", null),
        Arguments.of("feature only", Map.of("feature", "summarization")),
        Arguments.of("another feature", Map.of("feature", "reranking")),
        Arguments.of(
            "summarization/llm_id",
            Map.of("feature", "summarization", "required_param", "llm_id")));
  }

  @ParameterizedTest(name = "{0}")
  @MethodSource("featureDisabledDetails")
  void featureDisabledIsNoiseWhateverItsDetails(String label, Map<String, ?> details) {
    String disabled = status("FEATURE_DISABLED", "Feature disabled", details);
    events(disabled, definition(MEMORY, SPACE, Map.of()), chunk(CHUNK, MEMORY, "Kept", -0.5));
    var captured = captureStderr(() -> retriever().build().retrieve(Query.from("question")));
    var metadata = captured.value().getFirst().textSegment().metadata();
    assertFalse(metadata.containsKey("goodmem_partial"), label);
    assertFalse(metadata.containsKey("goodmem_statuses"), label);
    assertEquals(0, captured.warnings(), captured.stderr());

    events(disabled, definition(MEMORY, SPACE, Map.of()), chunk(CHUNK, MEMORY, "Kept", -0.5));
    var strict = retriever().failOnIncompleteRetrieval(true).build();
    assertEquals(List.of("Kept"), texts(strict.retrieve(Query.from("question"))), label);

    events(disabled, definition(MEMORY, SPACE, Map.of()), chunk(CHUNK, MEMORY, "Kept", -0.5));
    var admin =
        new GoodMemTools(client)
            .goodmemRetrieveMemories("question", List.of(SPACE), 5, null, null, null, null);
    assertFalse(admin.partial(), label);
  }

  @Test
  void rerankerFailureUnderAnotherCodeIsLabelledVectorFromTheStageName() throws Exception {
    events(
        status("RATE_LIMITED", "reranker provider rate limited; returning vector order", null),
        boundary("BEGIN", "retrieve"),
        definition(MEMORY, SPACE, Map.of()),
        chunk(CHUNK, MEMORY, "Vector ordered", -0.5),
        boundary("END", "retrieve"));
    var content =
        retriever().rerankerId(RERANKER).build().retrieve(Query.from("question")).getFirst();
    assertEquals("vector", content.textSegment().metadata().getString("goodmem_score_type"));
    assertNull(content.metadata().get(ContentMetadata.RERANKED_SCORE));
    assertEquals(-0.5, content.metadata().get(ContentMetadata.SCORE));
  }

  @Test
  void rerankStageKeepsRerankerScoresDespiteAnUnrelatedNotFound() throws Exception {
    events(
        status("NOT_FOUND", "Space filter referenced a missing field", null),
        boundary("BEGIN", "rerank"),
        definition(MEMORY, SPACE, Map.of()),
        chunk(CHUNK, MEMORY, "Reranked", 0.93),
        boundary("END", "rerank"));
    var content =
        retriever().rerankerId(RERANKER).build().retrieve(Query.from("question")).getFirst();
    assertEquals("reranker", content.textSegment().metadata().getString("goodmem_score_type"));
    assertEquals(0.93, content.metadata().get(ContentMetadata.RERANKED_SCORE));
    assertEquals("true", content.textSegment().metadata().getString("goodmem_partial"));
  }

  @Test
  void withoutStageNamesAFailedRerankerStillMeansVectorScores() throws Exception {
    events(
        status("RERANKING_FAILED", "Reranking skipped", null),
        definition(MEMORY, SPACE, Map.of()),
        chunk(CHUNK, MEMORY, "Vector ordered", -0.5));
    var content =
        retriever().rerankerId(RERANKER).build().retrieve(Query.from("question")).getFirst();
    assertEquals("vector", content.textSegment().metadata().getString("goodmem_score_type"));
    assertNull(content.metadata().get(ContentMetadata.RERANKED_SCORE));
  }
}
