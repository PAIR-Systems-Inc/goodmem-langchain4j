package ai.pairsys.goodmem.langchain4j;

import ai.pairsys.goodmem.client.ChatPostProcessorConfig;
import ai.pairsys.goodmem.client.Goodmem;
import ai.pairsys.goodmem.client.models.RerankerId;
import ai.pairsys.goodmem.client.models.RetrieveMemoryRequest;
import ai.pairsys.goodmem.client.models.SpaceId;
import ai.pairsys.goodmem.client.models.SpaceKey;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.service.tool.AiServiceTool;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

/**
 * Retrieves source-bearing content for LangChain4j RAG workflows. The caller owns the SDK client.
 * Search scope and optional metadata filters are developer configured. Blocking retrieval can be
 * used with LangChain4j's explicit {@code offloadBlocking(true)} option.
 */
public final class GoodMemContentRetriever implements ContentRetriever {
  private final Goodmem client;
  private final List<SpaceId> spaceIds;
  private final int maxResults;
  private final int fetchK;
  private final String filter;
  private final Function<Query, String> dynamicFilter;
  private final RerankerId rerankerId;

  private GoodMemContentRetriever(Builder builder) {
    client = Objects.requireNonNull(builder.client, "client");
    spaceIds = List.copyOf(Objects.requireNonNull(builder.spaceIds, "spaceIds"));
    if (spaceIds.isEmpty()) {
      throw new IllegalArgumentException("At least one spaceId is required");
    }
    maxResults = positive(builder.maxResults, "maxResults");
    rerankerId = builder.rerankerId;
    fetchK =
        builder.fetchK == null
            ? (rerankerId == null ? maxResults : Math.max(20, maxResults))
            : positive(builder.fetchK, "fetchK");
    if (fetchK < maxResults) {
      throw new IllegalArgumentException("fetchK must be at least maxResults");
    }
    filter = builder.filter;
    dynamicFilter = builder.dynamicFilter;
  }

  /**
   * Creates a builder for an application-scoped retriever.
   *
   * @return a new builder
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Returns matching content in server order, or throws if retrieval was incomplete.
   *
   * @param query query text and optional application metadata
   * @return matching chunks with source metadata; empty for a successful search with no matches
   * @throws GoodMemRetrievalException if the server reports incomplete retrieval
   */
  @Override
  public List<Content> retrieve(Query query) {
    Objects.requireNonNull(query, "query");
    String effectiveFilter = filter;
    if (dynamicFilter != null) {
      String dynamic = dynamicFilter.apply(query);
      if (dynamic != null && !dynamic.isBlank()) {
        effectiveFilter =
            filter == null || filter.isBlank() ? dynamic : "(" + filter + ") AND (" + dynamic + ")";
      }
    }
    final String selectedFilter = effectiveFilter;
    var request =
        RetrieveMemoryRequest.builder()
            .message(query.text())
            .spaceKeys(spaceIds.stream().map(id -> new SpaceKey(id, null, selectedFilter)).toList())
            .requestedSize(fetchK)
            .fetchMemory(true)
            .fetchMemoryContent(false);
    if (rerankerId != null) {
      request.postProcessor(
          ChatPostProcessorConfig.builder()
              .rerankerId(rerankerId)
              .maxResults((long) maxResults)
              .build());
    }
    return RetrievalResults.content(
        RetrievalResults.read(client, request.build()),
        Set.copyOf(spaceIds),
        maxResults,
        rerankerId != null);
  }

  /**
   * Returns an agent search tool whose only model-visible input is a query.
   *
   * @return a tool bound to this retriever's configuration
   */
  public AiServiceTool asTool() {
    return asTool(
        "goodmemSearch",
        "Search the configured GoodMem knowledge sources for relevant text and sources.");
  }

  /**
   * Creates a named search tool for use with {@code AiServices.tools(List<AiServiceTool>)}.
   *
   * @param name unique tool name, using letters, digits, underscores or hyphens
   * @param description description of these sources and when the agent should search them
   * @return a native LangChain4j tool bound to this retriever
   */
  public AiServiceTool asTool(String name, String description) {
    return GoodMemSearchTool.create(this, name, description);
  }

  static int positive(int value, String name) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be positive");
    }
    return value;
  }

  /** Configuration for retrieval; credentials and HTTP settings belong to the SDK client. */
  public static final class Builder {
    private Goodmem client;
    private List<SpaceId> spaceIds;
    private int maxResults = 5;
    private Integer fetchK;
    private String filter;
    private Function<Query, String> dynamicFilter;
    private RerankerId rerankerId;

    private Builder() {}

    /**
     * Sets the caller-owned GoodMem SDK client.
     *
     * @param client initialized SDK client
     * @return this builder
     */
    public Builder client(Goodmem client) {
      this.client = client;
      return this;
    }

    /**
     * Sets the spaces this retriever can search.
     *
     * @param spaceIds one or more space UUIDs
     * @return this builder
     */
    public Builder spaceIds(List<String> spaceIds) {
      this.spaceIds = spaceIds.stream().map(SpaceId::from).toList();
      return this;
    }

    /**
     * Sets the maximum number of returned chunks; defaults to five.
     *
     * @param maxResults positive result limit
     * @return this builder
     */
    public Builder maxResults(int maxResults) {
      this.maxResults = maxResults;
      return this;
    }

    /**
     * Sets candidate count before reranking; defaults to at least twenty when reranking is enabled.
     *
     * @param fetchK candidate count, at least {@code maxResults}
     * @return this builder
     */
    public Builder fetchK(int fetchK) {
      this.fetchK = fetchK;
      return this;
    }

    /**
     * Applies a native GoodMem metadata filter to every configured space.
     *
     * @param filter filter expression, or null for no fixed filter
     * @return this builder
     */
    public Builder filterExpression(String filter) {
      this.filter = filter;
      return this;
    }

    /**
     * Adds a native GoodMem expression per query, combined with the fixed expression using AND. Use
     * {@link GoodMemFilters#textEquals(String, String)} for application-supplied text values.
     *
     * @param dynamicFilter function returning a filter expression, or null to disable
     * @return this builder
     */
    public Builder dynamicFilterExpression(Function<Query, String> dynamicFilter) {
      this.dynamicFilter = dynamicFilter;
      return this;
    }

    /**
     * Enables server-side reranking without configuring an LLM.
     *
     * @param rerankerId reranker UUID, or null to disable reranking
     * @return this builder
     */
    public Builder rerankerId(String rerankerId) {
      this.rerankerId = rerankerId == null ? null : RerankerId.from(rerankerId);
      return this;
    }

    /**
     * Builds the configured retriever.
     *
     * @return a retriever with validated configuration
     */
    public GoodMemContentRetriever build() {
      return new GoodMemContentRetriever(this);
    }
  }
}
