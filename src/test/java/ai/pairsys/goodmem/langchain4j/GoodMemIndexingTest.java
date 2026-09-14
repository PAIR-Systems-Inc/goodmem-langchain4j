package ai.pairsys.goodmem.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.pairsys.goodmem.client.models.MemoryId;
import ai.pairsys.goodmem.client.models.MemoryProcessingStatus;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class GoodMemIndexingTest extends SdkTestSupport {
  @Test
  void largeReadinessChecksUseBatchesAndOnlyRecheckPendingIds() throws Exception {
    var ids =
        java.util.stream.IntStream.range(0, 250)
            .mapToObj(
                i ->
                    ai.pairsys.goodmem.client.models.MemoryId.from(
                        new java.util.UUID(0, i + 1).toString()))
            .toList();
    for (int start = 0; start < ids.size(); start += 100) {
      reply(
          batchMemories(
              ids.subList(start, Math.min(start + 100, ids.size())).stream()
                  .map(
                      id ->
                          memory(
                              id.toString(),
                              SPACE,
                              id.equals(ids.getLast()) ? "PENDING" : "COMPLETED",
                              Map.of()))
                  .toList()));
    }
    reply(batchMemories(List.of(memory(ids.getLast().toString(), SPACE, "COMPLETED", Map.of()))));
    GoodMemIndexing.waitForMemories(client, ids, Duration.ofSeconds(5));
    for (int i = 0; i < 3; i++) {
      var sent = request();
      assertEquals("/v1/memories:batchGet", sent.getPath());
      var payload = body(sent);
      assertEquals(i == 2 ? 50 : 100, payload.get("memoryIds").size());
      assertFalse(payload.path("includeContent").asBoolean());
    }
    assertEquals(ids.getLast().toString(), body(request()).at("/memoryIds/0").asText());
    assertEquals(4, server.getRequestCount());
  }

  @Test
  void timeoutPreservesAcceptedAndPendingIdsAndWaitingCanResume() {
    var ids = List.of(MemoryId.from(MEMORY), MemoryId.from(MEMORY_2));
    reply(
        batchMemories(
            List.of(
                memory(MEMORY, SPACE, "COMPLETED", Map.of()),
                memory(MEMORY_2, SPACE, "PENDING", Map.of()))));
    var failure =
        assertThrows(
            GoodMemIndexingException.class,
            () -> GoodMemIndexing.waitForMemories(client, ids, Duration.ofMillis(100)));
    assertEquals(ids, failure.memoryIds());
    assertEquals(List.of(MemoryId.from(MEMORY_2)), failure.pendingMemoryIds());
    reply(batchMemories(List.of(memory(MEMORY_2, SPACE, "COMPLETED", Map.of()))));
    GoodMemIndexing.waitForMemories(client, failure.pendingMemoryIds(), Duration.ofSeconds(1));
    assertEquals(2, server.getRequestCount());
  }

  @Test
  void anUnrelatedOrMissingMemoryCannotSatisfyTheWait() {
    reply(batchMemories(List.of(memory(MEMORY_2, SPACE, "COMPLETED", Map.of()))));
    var failure =
        assertThrows(
            GoodMemIndexingException.class,
            () ->
                GoodMemIndexing.waitForMemories(
                    client, List.of(MemoryId.from(MEMORY)), Duration.ofSeconds(1)));
    assertEquals(List.of(MemoryId.from(MEMORY)), failure.pendingMemoryIds());
  }

  @Test
  void pollsTheWrittenMemoryUntilCompleted() throws Exception {
    for (String state : new String[] {"PENDING", "PROCESSING", "COMPLETED"}) {
      reply(json(memory(MEMORY, SPACE, state, Map.of())));
    }
    var result =
        GoodMemIndexing.waitForMemory(client, MemoryId.from(MEMORY), Duration.ofSeconds(2));
    assertEquals(MemoryProcessingStatus.COMPLETED, result.processingStatus());
    assertEquals(3, server.getRequestCount());
    for (int i = 0; i < 3; i++) {
      var sent = request();
      assertEquals("GET", sent.getMethod());
      assertEquals("/v1/memories/" + MEMORY, sent.getPath());
    }
  }

  @Test
  void failedIndexingStopsImmediately() {
    reply(json(memory(MEMORY, SPACE, "FAILED", Map.of())));
    var failure =
        assertThrows(
            GoodMemException.class,
            () ->
                GoodMemIndexing.waitForMemory(
                    client, MemoryId.from(MEMORY), Duration.ofSeconds(2)));
    assertTrue(failure.getMessage().contains(MEMORY));
    assertTrue(failure.getMessage().contains("FAILED"));
    assertEquals(1, server.getRequestCount());
  }

  @Test
  void stopsPollingAtTheDeadline() {
    reply(json(memory(MEMORY, SPACE, "PENDING", Map.of())));
    var failure =
        assertThrows(
            GoodMemException.class,
            () ->
                GoodMemIndexing.waitForMemory(
                    client, MemoryId.from(MEMORY), Duration.ofMillis(100)));
    assertTrue(failure.getMessage().contains("Timed out"));
    assertEquals(1, server.getRequestCount());
  }

  @Test
  void interruptionPreservesTheInterruptFlagAndDoesNotStartHttp() {
    Thread.currentThread().interrupt();
    try {
      assertThrows(
          GoodMemException.class,
          () ->
              GoodMemIndexing.waitForMemory(client, MemoryId.from(MEMORY), Duration.ofSeconds(1)));
      assertTrue(Thread.currentThread().isInterrupted());
      assertEquals(0, server.getRequestCount());
    } finally {
      Thread.interrupted();
    }
  }
}
