package ai.pairsys.goodmem.langchain4j;

import ai.pairsys.goodmem.client.Goodmem;
import ai.pairsys.goodmem.client.models.BatchMemoryRetrievalRequest;
import ai.pairsys.goodmem.client.models.Memory;
import ai.pairsys.goodmem.client.models.MemoryId;
import ai.pairsys.goodmem.client.models.MemoryProcessingStatus;
import java.time.Duration;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;

/** Readiness checks for specific memories written through the SDK or this integration. */
public final class GoodMemIndexing {
  private GoodMemIndexing() {}

  /**
   * Waits for the specified memories using batch status reads. Completed IDs leave the polling set.
   *
   * @param client caller-owned SDK client
   * @param memoryIds accepted memory IDs to check; no other memories are queried
   * @param timeout explicit polling budget; SDK request timeouts apply to each HTTP call
   * @throws GoodMemIndexingException when readiness cannot be confirmed, with IDs for a later retry
   */
  public static void waitForMemories(Goodmem client, List<MemoryId> memoryIds, Duration timeout) {
    Objects.requireNonNull(client, "client");
    List<MemoryId> ids = List.copyOf(memoryIds);
    positive(timeout);
    var pending = new LinkedHashSet<>(ids);
    long started = System.nanoTime();
    try {
      while (!pending.isEmpty()) {
        var round = List.copyOf(pending);
        for (int offset = 0; offset < round.size(); offset += 100) {
          checkWait(started, timeout);
          var batch = round.subList(offset, Math.min(offset + 100, round.size()));
          var response =
              client.memories.batchGet(
                  BatchMemoryRetrievalRequest.builder()
                      .memoryIds(batch)
                      .includeContent(false)
                      .includeProcessingHistory(false)
                      .build());
          var unseen = new LinkedHashSet<>(batch);
          if (response.results() == null) {
            throw new GoodMemException("Indexing status response omitted results");
          }
          for (var result : response.results()) {
            Memory memory = result.memory();
            if (!Boolean.TRUE.equals(result.success()) || memory == null) {
              throw new GoodMemException("Cannot read indexing status: " + result.error());
            }
            if (!unseen.remove(memory.memoryId())) {
              throw new GoodMemException("Unexpected or duplicate memory in indexing response");
            }
            if (memory.processingStatus() == MemoryProcessingStatus.COMPLETED) {
              pending.remove(memory.memoryId());
            } else if (memory.processingStatus() != MemoryProcessingStatus.PENDING
                && memory.processingStatus() != MemoryProcessingStatus.PROCESSING) {
              throw new GoodMemException(
                  "Memory "
                      + memory.memoryId()
                      + " has processing status "
                      + memory.processingStatus());
            }
          }
          if (!unseen.isEmpty()) {
            throw new GoodMemException("Indexing response omitted requested memories: " + unseen);
          }
        }
        if (!pending.isEmpty()) {
          checkWait(started, timeout);
          long remaining = timeout.toNanos() - (System.nanoTime() - started);
          try {
            Thread.sleep(
                Duration.ofNanos(
                    Math.max(1, Math.min(remaining, Duration.ofMillis(200).toNanos()))));
          } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new GoodMemException("Indexing wait interrupted", e);
          }
        }
      }
    } catch (RuntimeException e) {
      throw new GoodMemIndexingException(
          "Indexing readiness not confirmed: " + e.getMessage(), ids, List.copyOf(pending), e);
    }
  }

  private static void checkWait(long started, Duration timeout) {
    if (Thread.currentThread().isInterrupted()) {
      throw new GoodMemException("Indexing wait interrupted");
    }
    if (System.nanoTime() - started >= timeout.toNanos()) {
      throw new GoodMemException("Indexing wait timed out");
    }
  }

  /**
   * Waits for one memory to finish indexing, without performing a search.
   *
   * @param client caller-owned SDK client
   * @param memoryId memory whose processing status should be checked
   * @param timeout maximum polling time; SDK request timeouts apply to each HTTP call
   * @return the completed memory
   * @throws GoodMemException when processing fails, polling times out, or the thread is interrupted
   */
  public static Memory waitForMemory(Goodmem client, MemoryId memoryId, Duration timeout) {
    Objects.requireNonNull(client, "client");
    Objects.requireNonNull(memoryId, "memoryId");
    positive(timeout);
    long started = System.nanoTime();
    while (true) {
      if (Thread.currentThread().isInterrupted()) {
        throw new GoodMemException("Interrupted while waiting for memory " + memoryId);
      }
      if (System.nanoTime() - started >= timeout.toNanos()) {
        throw new GoodMemException("Timed out waiting for memory " + memoryId);
      }
      Memory memory = client.memories.get(memoryId);
      if (memory.processingStatus() == MemoryProcessingStatus.COMPLETED) {
        return memory;
      }
      if (memory.processingStatus() != MemoryProcessingStatus.PENDING
          && memory.processingStatus() != MemoryProcessingStatus.PROCESSING) {
        throw new GoodMemException(
            "Memory " + memoryId + " has processing status " + memory.processingStatus());
      }
      long remaining = timeout.toNanos() - (System.nanoTime() - started);
      if (remaining <= 0) {
        throw new GoodMemException("Timed out waiting for memory " + memoryId);
      }
      try {
        Thread.sleep(Duration.ofNanos(Math.min(remaining, Duration.ofMillis(200).toNanos())));
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new GoodMemException("Interrupted while waiting for memory " + memoryId, e);
      }
    }
  }

  static Duration positive(Duration timeout) {
    Objects.requireNonNull(timeout, "indexingTimeout");
    if (timeout.isNegative() || timeout.isZero()) {
      throw new IllegalArgumentException("indexingTimeout must be positive");
    }
    timeout.toNanos();
    return timeout;
  }
}
