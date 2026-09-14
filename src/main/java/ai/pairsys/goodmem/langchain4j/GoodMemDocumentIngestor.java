package ai.pairsys.goodmem.langchain4j;

import ai.pairsys.goodmem.client.Goodmem;
import ai.pairsys.goodmem.client.models.JsonBatchMemoryCreationRequest;
import ai.pairsys.goodmem.client.models.JsonMemoryCreationRequest;
import ai.pairsys.goodmem.client.models.MemoryId;
import ai.pairsys.goodmem.client.models.SpaceId;
import dev.langchain4j.data.document.Document;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Writes whole LangChain4j Documents; GoodMem performs chunking and embedding. */
public final class GoodMemDocumentIngestor {
  private final Goodmem client;
  private final SpaceId spaceId;
  private static final int BATCH_SIZE = 100;

  private GoodMemDocumentIngestor(Builder builder) {
    client = Objects.requireNonNull(builder.client, "client");
    spaceId = Objects.requireNonNull(builder.spaceId, "spaceId");
  }

  /**
   * Creates an ingestor builder.
   *
   * @return a new builder
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Writes documents in batches and returns their accepted IDs without waiting for indexing.
   *
   * @param documents documents whose text and metadata should be stored
   * @return memory IDs in input order
   * @throws GoodMemIngestionException if a batch fails; confirmed writes remain stored
   */
  public List<MemoryId> ingest(List<Document> documents) {
    Objects.requireNonNull(documents, "documents");
    List<JsonMemoryCreationRequest> requests =
        documents.stream()
            .map(
                document -> {
                  Objects.requireNonNull(document, "document");
                  var metadata = document.metadata().toMap();
                  Object source = metadata.get("source");
                  return JsonMemoryCreationRequest.builder()
                      .spaceId(spaceId)
                      .originalContent(document.text())
                      .metadata(metadata)
                      .contentType("text/plain")
                      .originalContentRef(source instanceof String s ? s : null)
                      .build();
                })
            .toList();
    List<MemoryId> created = new ArrayList<>();
    try {
      for (int offset = 0; offset < requests.size(); offset += BATCH_SIZE) {
        List<JsonMemoryCreationRequest> batch =
            requests.subList(offset, Math.min(offset + BATCH_SIZE, requests.size()));
        var response =
            client.memories.batchCreate(
                JsonBatchMemoryCreationRequest.builder().requests(batch).build());
        if (response.results() == null) {
          throw new GoodMemException("Batch response omitted item results");
        }
        MemoryId[] ordered = new MemoryId[batch.size()];
        List<String> failures = new ArrayList<>();
        for (int position = 0; position < response.results().size(); position++) {
          var result = response.results().get(position);
          if (Boolean.TRUE.equals(result.success())
              && result.memory() != null
              && result.memory().memoryId() != null) {
            MemoryId id = result.memory().memoryId();
            created.add(id);
            long index = result.requestIndex() == null ? position : result.requestIndex();
            if (index < 0 || index >= ordered.length || ordered[(int) index] != null) {
              failures.add("Invalid or duplicate batch requestIndex: " + index);
            } else {
              ordered[(int) index] = id;
            }
          } else {
            failures.add("Batch item " + position + " failed: " + result.error());
          }
        }
        if (!failures.isEmpty()
            || java.util.Arrays.asList(ordered).contains(null)
            || response.results().size() != batch.size()) {
          throw new GoodMemException("Incomplete batch write: " + failures);
        }
        created.subList(created.size() - ordered.length, created.size()).clear();
        created.addAll(List.of(ordered));
      }
      return List.copyOf(created);
    } catch (RuntimeException e) {
      if (created.isEmpty() && e instanceof ai.pairsys.goodmem.client.errors.GoodmemException) {
        throw e;
      }
      throw new GoodMemIngestionException(
          "Document ingestion failed: " + e.getMessage(), created, e);
    }
  }

  /**
   * Stores documents, then waits for readiness within an explicit polling budget.
   *
   * @param documents documents to store
   * @param timeout maximum readiness polling time after acceptance; SDK request timeouts apply
   *     separately
   * @return accepted memory IDs, now indexed, in input order
   * @throws GoodMemIngestionException if writing fails
   * @throws GoodMemIndexingException if accepted writes have not all finished indexing; use its IDs
   *     to resume waiting
   */
  public List<MemoryId> ingestAndWait(List<Document> documents, Duration timeout) {
    GoodMemIndexing.positive(timeout);
    List<MemoryId> ids = ingest(documents);
    GoodMemIndexing.waitForMemories(client, ids, timeout);
    return ids;
  }

  /** Configuration shared by all documents written through this ingestor. */
  public static final class Builder {
    private Goodmem client;
    private SpaceId spaceId;

    private Builder() {}

    /**
     * Sets the caller-owned SDK client.
     *
     * @param client initialized SDK client
     * @return this builder
     */
    public Builder client(Goodmem client) {
      this.client = client;
      return this;
    }

    /**
     * Sets the destination space.
     *
     * @param spaceId destination space UUID
     * @return this builder
     */
    public Builder spaceId(String spaceId) {
      this.spaceId = SpaceId.from(spaceId);
      return this;
    }

    /**
     * Builds the configured ingestor.
     *
     * @return an ingestor with validated configuration
     */
    public GoodMemDocumentIngestor build() {
      return new GoodMemDocumentIngestor(this);
    }
  }
}
