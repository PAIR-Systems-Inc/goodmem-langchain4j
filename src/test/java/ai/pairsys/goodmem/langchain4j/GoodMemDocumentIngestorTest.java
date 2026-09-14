package ai.pairsys.goodmem.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.pairsys.goodmem.client.models.MemoryId;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.IntStream;
import okhttp3.mockwebserver.MockResponse;
import org.junit.jupiter.api.Test;

class GoodMemDocumentIngestorTest extends SdkTestSupport {
  GoodMemDocumentIngestor.Builder ingestor() {
    return GoodMemDocumentIngestor.builder().client(client).spaceId(SPACE);
  }

  @Test
  void writesWholeDocumentsAndMetadataThenWaitsAndReturnsIdsInInputOrder() throws Exception {
    reply(
        json(
            Map.of(
                "results",
                List.of(
                    Map.of(
                        "success",
                        true,
                        "requestIndex",
                        1,
                        "memory",
                        memory(MEMORY_2, SPACE, "PENDING", Map.of())),
                    Map.of(
                        "success",
                        true,
                        "requestIndex",
                        0,
                        "memory",
                        memory(MEMORY, SPACE, "PENDING", Map.of()))))));
    reply(
        batchMemories(
            List.of(
                memory(MEMORY, SPACE, "COMPLETED", Map.of()),
                memory(MEMORY_2, SPACE, "COMPLETED", Map.of()))));
    var ids =
        ingestor()
            .build()
            .ingestAndWait(
                List.of(
                    Document.from(
                        "Full first document",
                        Metadata.from(
                            Map.of("source", "https://example.com/first", "revision", 3))),
                    Document.from("Second document")),
                Duration.ofSeconds(5));
    assertEquals(List.of(MemoryId.from(MEMORY), MemoryId.from(MEMORY_2)), ids);
    var sent = request();
    assertEquals("/v1/memories:batchCreate", sent.getPath());
    var documents = body(sent).get("requests");
    assertEquals("Full first document", documents.at("/0/originalContent").asText());
    assertEquals("https://example.com/first", documents.at("/0/originalContentRef").asText());
    assertEquals(3, documents.at("/0/metadata/revision").asInt());
    assertEquals(SPACE, documents.at("/0/spaceId").asText());
    assertFalse(documents.get(0).has("chunkingConfig"));
    assertEquals("/v1/memories:batchGet", request().getPath());
    assertEquals(2, server.getRequestCount());
  }

  @Test
  void partialWritePreservesAllConfirmedIdsAndDoesNotPretendToSucceed() {
    reply(
        json(
            Map.of(
                "results",
                List.of(
                    Map.of(
                        "success", false, "error", Map.of("code", 400, "message", "bad document")),
                    Map.of(
                        "success",
                        true,
                        "memory",
                        memory(MEMORY_2, SPACE, "PENDING", Map.of()))))));
    var failure =
        assertThrows(
            GoodMemIngestionException.class,
            () -> ingestor().build().ingest(List.of(Document.from("one"), Document.from("two"))));
    assertEquals(List.of(MemoryId.from(MEMORY_2)), failure.createdMemoryIds());
    assertTrue(failure.getMessage().contains("bad document"));
    assertEquals(1, server.getRequestCount());
  }

  @Test
  void indexingFailureIncludesIdsOfWrittenDocuments() {
    reply(
        json(
            Map.of(
                "results",
                List.of(
                    Map.of(
                        "success", true, "memory", memory(MEMORY, SPACE, "PENDING", Map.of()))))));
    reply(batchMemories(List.of(memory(MEMORY, SPACE, "FAILED", Map.of()))));
    var failure =
        assertThrows(
            GoodMemIndexingException.class,
            () ->
                ingestor()
                    .build()
                    .ingestAndWait(List.of(Document.from("one")), Duration.ofSeconds(5)));
    assertEquals(List.of(MemoryId.from(MEMORY)), failure.memoryIds());
    assertTrue(failure.getMessage().contains("FAILED"));
  }

  @Test
  void firstBatchHttpFailurePreservesTheSdkException() {
    server.enqueue(
        new MockResponse().setResponseCode(401).setBody("{\"message\":\"invalid key\"}"));
    assertThrows(
        ai.pairsys.goodmem.client.errors.AuthenticationException.class,
        () -> ingestor().build().ingest(List.of(Document.from("one"))));
    assertEquals(1, server.getRequestCount());
  }

  @Test
  void laterBatchFailureRetainsEarlierWritesAndDoesNotRetryTheWrite() {
    List<Map<String, Object>> results = new ArrayList<>();
    for (int i = 0; i < 100; i++) {
      results.add(
          Map.of(
              "success",
              true,
              "memory",
              memory(new UUID(0, i + 1).toString(), SPACE, "PENDING", Map.of())));
    }
    reply(json(Map.of("results", results)));
    server.enqueue(
        new MockResponse().setResponseCode(400).setBody("{\"message\":\"batch rejected\"}"));
    var documents = IntStream.range(0, 101).mapToObj(i -> Document.from("Document " + i)).toList();
    var failure =
        assertThrows(GoodMemIngestionException.class, () -> ingestor().build().ingest(documents));
    assertEquals(100, failure.createdMemoryIds().size());
    assertInstanceOf(
        ai.pairsys.goodmem.client.errors.BadRequestException.class, failure.getCause());
    assertEquals(2, server.getRequestCount());
  }

  @Test
  void ingestionReturnsAcceptedIdsAndEmptyInputMakesNoRequest() {
    var ingestor = ingestor().build();
    assertEquals(List.of(), ingestor.ingest(List.of()));
    assertEquals(0, server.getRequestCount());
    reply(
        json(
            Map.of(
                "results",
                List.of(
                    Map.of(
                        "success", true, "memory", memory(MEMORY, SPACE, "PENDING", Map.of()))))));
    assertEquals(List.of(MemoryId.from(MEMORY)), ingestor.ingest(List.of(Document.from("one"))));
    assertEquals(1, server.getRequestCount());
  }

  @Test
  void missingBatchResultsAreNotReportedAsCompleteIngestion() {
    reply(
        json(
            Map.of(
                "results",
                List.of(
                    Map.of(
                        "success", true, "memory", memory(MEMORY, SPACE, "PENDING", Map.of()))))));
    var failure =
        assertThrows(
            GoodMemIngestionException.class,
            () -> ingestor().build().ingest(List.of(Document.from("one"), Document.from("two"))));
    assertEquals(List.of(MemoryId.from(MEMORY)), failure.createdMemoryIds());
  }
}
