package ai.pairsys.goodmem.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.pairsys.goodmem.client.Goodmem;
import ai.pairsys.goodmem.client.models.GoodMemStatusCode;
import ai.pairsys.goodmem.client.models.MemoriesListStatusFilter;
import ai.pairsys.goodmem.client.models.Memory;
import ai.pairsys.goodmem.client.models.MemoryProcessingStatus;
import ai.pairsys.goodmem.client.models.Space;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.rag.query.Query;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Opt-in live checks. Each test deletes only the uniquely named space it created. */
class GoodMemLiveIT {
  private static Goodmem client;
  private static String embedderId;
  private static String rerankerId;
  private static final String BAD_ID = "00000000-0000-0000-0000-000000000000";

  @BeforeAll
  static void connect() {
    embedderId = required("GOODMEM_EMBEDDER_ID");
    rerankerId = required("GOODMEM_RERANKER_ID");
    client =
        Goodmem.builder()
            .baseUrl(required("GOODMEM_BASE_URL"))
            .apiKey(required("GOODMEM_API_KEY"))
            .timeout(Duration.ofSeconds(30))
            .build();
  }

  @AfterAll
  static void close() {
    if (client != null) {
      client.close();
    }
  }

  @Test
  void documentIngestionFilteringSourceMetadataAndReranking() {
    Space space = newSpace();
    try {
      var ingestor =
          GoodMemDocumentIngestor.builder()
              .client(client)
              .spaceId(space.spaceId().toString())
              .build();
      var ids =
          ingestor.ingestAndWait(
              List.of(
                  Document.from(
                      "The Meridian project's launch code is ORCHID. This is the launch operations guide.",
                      Metadata.from(
                          Map.of(
                              "project",
                              "meridian",
                              "source",
                              "https://example.com/meridian",
                              "title",
                              "Meridian guide"))),
                  Document.from(
                      "The Atlas project's launch code is CEDAR. This is the unrelated Atlas launch guide.",
                      Metadata.from(
                          Map.of("project", "atlas", "source", "https://example.com/atlas")))),
              Duration.ofSeconds(120));
      assertEquals(2, ids.size());
      String filter = "CAST(val('$.project') AS TEXT) = 'meridian'";
      for (String reranker : new String[] {null, rerankerId}) {
        var retriever =
            GoodMemContentRetriever.builder()
                .client(client)
                .spaceIds(List.of(space.spaceId().toString()))
                .filterExpression(filter)
                .rerankerId(reranker)
                .maxResults(2)
                .build();
        var contents =
            retriever.retrieve(Query.from("What is the Meridian project's launch code?"));
        assertFalse(contents.isEmpty());
        assertTrue(
            contents.stream()
                .allMatch(c -> c.textSegment().metadata().getString("project").equals("meridian")));
        assertTrue(contents.stream().anyMatch(c -> c.textSegment().text().contains("ORCHID")));
        assertEquals(
            "https://example.com/meridian",
            contents.getFirst().textSegment().metadata().getString("source"));
        assertEquals(
            ids.getFirst().toString(),
            contents.getFirst().textSegment().metadata().getString("memory_id"));
      }
      var empty =
          GoodMemContentRetriever.builder()
              .client(client)
              .spaceIds(List.of(space.spaceId().toString()))
              .filterExpression("CAST(val('$.project') AS TEXT) = 'absent'")
              .build();
      assertTrue(empty.retrieve(Query.from("launch code")).isEmpty());
    } finally {
      client.spaces.delete(space.spaceId());
    }
  }

  @Test
  void filterValuesStayDataEvenWithQuotesBackslashesAndNewlines() {
    Space space = newSpace();
    try {
      String tenant = "O'Reilly\\notes\n' OR TRUE OR 'x' = 'x";
      var ingestor =
          GoodMemDocumentIngestor.builder()
              .client(client)
              .spaceId(space.spaceId().toString())
              .build();
      ingestor.ingestAndWait(
          List.of(
              Document.from(
                  "The tenant's secret launch code is ORCHID.", Metadata.from("tenant", tenant)),
              Document.from(
                  "The other tenant's secret launch code is CEDAR.",
                  Metadata.from("tenant", "other"))),
          Duration.ofSeconds(120));
      var retriever =
          GoodMemContentRetriever.builder()
              .client(client)
              .spaceIds(List.of(space.spaceId().toString()))
              .dynamicFilterExpression(query -> GoodMemFilters.textEquals("tenant", tenant))
              .build();
      var results = retriever.retrieve(Query.from("secret launch code"));
      assertFalse(results.isEmpty());
      assertTrue(
          results.stream()
              .allMatch(c -> tenant.equals(c.textSegment().metadata().getString("tenant"))));
    } finally {
      client.spaces.delete(space.spaceId());
    }
  }

  @Test
  void invalidRerankerCannotMasqueradeAsSuccessfulReranking() {
    Space space = newSpace();
    try {
      var tools = new GoodMemTools(client);
      tools.goodmemCreateMemory(
          space.spaceId().toString(),
          "GoodMem stores reusable memories for Java applications.",
          null,
          null,
          null);
      var events =
          tools.goodmemRetrieveMemories(
              "Java memory", List.of(space.spaceId().toString()), 3, null, null, BAD_ID, null);
      assertTrue(
          events.statuses().stream()
              .anyMatch(
                  e ->
                      e.code() == GoodMemStatusCode.NOT_FOUND
                          || e.code() == GoodMemStatusCode.RERANKING_FAILED),
          "An invalid reranker must produce a visible diagnostic");
      var retriever =
          GoodMemContentRetriever.builder()
              .client(client)
              .spaceIds(List.of(space.spaceId().toString()))
              .rerankerId(BAD_ID)
              .build();
      assertThrows(
          GoodMemRetrievalException.class, () -> retriever.retrieve(Query.from("Java memory")));
    } finally {
      client.spaces.delete(space.spaceId());
    }
  }

  @Test
  void sdkToolsCreateReadUpdateListUploadAndDelete(@TempDir Path directory) throws Exception {
    Space space = newSpace();
    var tools = new GoodMemTools(client, Duration.ofSeconds(60), directory);
    try {
      String sid = space.spaceId().toString();
      assertTrue(
          tools.goodmemListEmbedders().stream()
              .anyMatch(e -> e.embedderId().toString().equals(embedderId)));
      assertEquals(space.spaceId(), tools.goodmemGetSpace(sid).spaceId());
      String newName = space.name() + "-updated";
      assertEquals(
          newName,
          tools.goodmemUpdateSpace(sid, newName, null, Map.of("test", "langchain4j")).name());
      assertTrue(
          tools.goodmemListSpaces(newName, 100).stream()
              .anyMatch(s -> s.spaceId().equals(space.spaceId())));
      Path file = directory.resolve("memory.txt");
      String text = "File uploads preserve the original document and its source metadata.";
      Files.writeString(file, text);
      Memory memory =
          tools.goodmemCreateMemory(
              sid, null, file.toString(), Map.of("source", "local-file"), null);
      assertEquals(MemoryProcessingStatus.COMPLETED, memory.processingStatus());
      assertEquals(text, tools.goodmemGetMemory(memory.memoryId().toString(), true).text());
      assertTrue(
          tools.goodmemListMemories(sid, 100, MemoriesListStatusFilter.COMPLETED, null).stream()
              .anyMatch(m -> m.memoryId().equals(memory.memoryId())));
      var found =
          GoodMemContentRetriever.builder()
              .client(client)
              .spaceIds(List.of(sid))
              .build()
              .retrieve(Query.from("original document source"));
      assertTrue(
          found.stream()
              .anyMatch(
                  c ->
                      c.textSegment()
                          .metadata()
                          .getString("memory_id")
                          .equals(memory.memoryId().toString())));
      tools.goodmemDeleteMemory(memory.memoryId().toString());
      assertTrue(tools.goodmemListMemories(sid, 100, null, null).isEmpty());
    } finally {
      tools.goodmemDeleteSpace(space.spaceId().toString());
    }
  }

  private static Space newSpace() {
    return new GoodMemTools(client)
        .goodmemCreateSpace(
            "langchain4j-0.2-" + UUID.randomUUID(),
            embedderId,
            Map.of("purpose", "integration-test"));
  }

  private static String required(String name) {
    String value = System.getenv(name);
    if (value == null || value.isBlank()) {
      throw new IllegalStateException("Set " + name + " before running -Plive");
    }
    return value;
  }
}
