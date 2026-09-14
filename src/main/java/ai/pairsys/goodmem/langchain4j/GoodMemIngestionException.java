package ai.pairsys.goodmem.langchain4j;

import ai.pairsys.goodmem.client.models.MemoryId;
import java.util.List;

/** A write or indexing failure, with IDs of the writes already confirmed by the server. */
public final class GoodMemIngestionException extends GoodMemException {
  /** Writes confirmed by the server before ingestion failed. */
  private final List<MemoryId> createdMemoryIds;

  GoodMemIngestionException(String message, List<MemoryId> createdMemoryIds, Throwable cause) {
    super(message, cause);
    this.createdMemoryIds = List.copyOf(createdMemoryIds);
  }

  /**
   * Returns confirmed writes; a transport failure can leave additional writes unconfirmed.
   *
   * @return an immutable list of confirmed memory IDs
   */
  public List<MemoryId> createdMemoryIds() {
    return createdMemoryIds;
  }
}
