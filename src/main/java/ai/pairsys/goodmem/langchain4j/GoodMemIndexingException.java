package ai.pairsys.goodmem.langchain4j;

import ai.pairsys.goodmem.client.models.MemoryId;
import java.util.List;

/** A readiness failure after acceptance; retry the wait using the IDs, without uploading again. */
public final class GoodMemIndexingException extends GoodMemException {
  /** All IDs supplied to the readiness check. */
  private final List<MemoryId> memoryIds;

  /** IDs whose completion was not confirmed. */
  private final List<MemoryId> pendingMemoryIds;

  GoodMemIndexingException(
      String message, List<MemoryId> memoryIds, List<MemoryId> pendingMemoryIds, Throwable cause) {
    super(message, cause);
    this.memoryIds = List.copyOf(memoryIds);
    this.pendingMemoryIds = List.copyOf(pendingMemoryIds);
  }

  /**
   * Returns all IDs supplied to the readiness check.
   *
   * @return accepted memory IDs, including any that already completed indexing
   */
  public List<MemoryId> memoryIds() {
    return memoryIds;
  }

  /**
   * Returns the IDs whose completion was not confirmed.
   *
   * @return IDs to inspect or use for a subsequent readiness check
   */
  public List<MemoryId> pendingMemoryIds() {
    return pendingMemoryIds;
  }
}
