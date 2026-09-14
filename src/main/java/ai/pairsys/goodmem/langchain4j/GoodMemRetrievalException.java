package ai.pairsys.goodmem.langchain4j;

import ai.pairsys.goodmem.client.models.GoodMemStatus;
import java.util.List;

/** Retrieval was incomplete; server diagnostics are available through {@link #statuses()}. */
public final class GoodMemRetrievalException extends GoodMemException {
  /** Server diagnostics that prevented complete retrieval. */
  private final List<GoodMemStatus> statuses;

  GoodMemRetrievalException(List<GoodMemStatus> statuses) {
    super("GoodMem retrieval was incomplete: " + statuses);
    this.statuses = List.copyOf(statuses);
  }

  /**
   * Returns the server statuses that prevented a complete retrieval.
   *
   * @return an immutable list of server diagnostics
   */
  public List<GoodMemStatus> statuses() {
    return statuses;
  }
}
