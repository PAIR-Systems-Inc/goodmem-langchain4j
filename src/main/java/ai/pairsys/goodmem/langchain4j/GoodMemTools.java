package ai.pairsys.goodmem.langchain4j;

import ai.pairsys.goodmem.client.ChatPostProcessorConfig;
import ai.pairsys.goodmem.client.Goodmem;
import ai.pairsys.goodmem.client.Page;
import ai.pairsys.goodmem.client.models.ChunkId;
import ai.pairsys.goodmem.client.models.EmbedderId;
import ai.pairsys.goodmem.client.models.EmbedderResponse;
import ai.pairsys.goodmem.client.models.GoodMemStatus;
import ai.pairsys.goodmem.client.models.JsonMemoryCreationRequest;
import ai.pairsys.goodmem.client.models.MemoriesListStatusFilter;
import ai.pairsys.goodmem.client.models.Memory;
import ai.pairsys.goodmem.client.models.MemoryGetOptions;
import ai.pairsys.goodmem.client.models.MemoryId;
import ai.pairsys.goodmem.client.models.MemoryListOptions;
import ai.pairsys.goodmem.client.models.MemoryProcessingStatus;
import ai.pairsys.goodmem.client.models.RetrieveMemoryRequest;
import ai.pairsys.goodmem.client.models.Space;
import ai.pairsys.goodmem.client.models.SpaceCreationRequest;
import ai.pairsys.goodmem.client.models.SpaceEmbedderConfig;
import ai.pairsys.goodmem.client.models.SpaceId;
import ai.pairsys.goodmem.client.models.SpaceKey;
import ai.pairsys.goodmem.client.models.SpaceListOptions;
import ai.pairsys.goodmem.client.models.UpdateSpaceRequest;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * SDK-backed administrative and memory tools for LangChain4j agents. Search and read operations
 * return compact, readable results; administrative methods return SDK models. LangChain4j
 * serializes tool results and handles exceptions. For application-scoped search, use {@link
 * GoodMemContentRetriever#asTool(String, String)}. Local file uploads are disabled unless the
 * application configures an upload directory.
 */
public final class GoodMemTools {
  private final Goodmem client;
  private final Duration indexingTimeout;
  private final Path uploadDirectory;

  /**
   * Creates tools with a caller-owned SDK client and a sixty-second indexing polling budget.
   *
   * @param client initialized SDK client
   */
  public GoodMemTools(Goodmem client) {
    this(client, Duration.ofSeconds(60));
  }

  /**
   * Creates tools with an explicit indexing polling budget. The caller owns and closes the client.
   *
   * @param client initialized SDK client
   * @param indexingTimeout positive polling budget; SDK request timeouts apply separately
   */
  public GoodMemTools(Goodmem client, Duration indexingTimeout) {
    this(client, indexingTimeout, null);
  }

  /**
   * Creates tools with optional file uploads restricted to a configured directory.
   *
   * @param client initialized SDK client
   * @param indexingTimeout positive indexing polling budget
   * @param uploadDirectory directory whose files may be uploaded; null disables local file access
   */
  public GoodMemTools(Goodmem client, Duration indexingTimeout, Path uploadDirectory) {
    this.client = Objects.requireNonNull(client, "client");
    this.indexingTimeout = GoodMemIndexing.positive(indexingTimeout);
    try {
      this.uploadDirectory = uploadDirectory == null ? null : uploadDirectory.toRealPath();
      if (this.uploadDirectory != null && !Files.isDirectory(this.uploadDirectory)) {
        throw new IllegalArgumentException("uploadDirectory must be a directory");
      }
    } catch (IOException e) {
      throw new UncheckedIOException("Cannot access the configured upload directory", e);
    }
  }

  /**
   * Creates a new space using SDK chunking defaults; existing spaces are never selected by name.
   *
   * @param name new space name
   * @param embedderId embedder UUID
   * @param labels optional string labels
   * @return the newly created space
   */
  @Tool(
      "Create a new GoodMem space using the specified embedder and server-compatible default chunking.")
  public Space goodmemCreateSpace(
      @P("Name for the new space") String name,
      @P("UUID of the embedder model") String embedderId,
      @P(value = "Optional string labels", required = false) Map<String, String> labels) {
    return client.spaces.create(
        SpaceCreationRequest.builder()
            .name(name)
            .labels(labels)
            .spaceEmbedders(List.of(new SpaceEmbedderConfig(EmbedderId.from(embedderId), 1.0)))
            .build());
  }

  /**
   * Stores text or a local file, preserving metadata and waiting for indexing by default.
   *
   * @param spaceId destination space UUID
   * @param textContent text to store, or null when uploading a file
   * @param filePath file within the configured upload directory, or null when storing text
   * @param metadata optional memory metadata
   * @param wait whether to wait for indexing; null means true
   * @return the created memory, indexed unless waiting is disabled
   */
  @Tool(
      "Store text in GoodMem, or upload a file if the application enabled an upload directory. Provide exactly one content input. Waits for indexing by default.")
  public Memory goodmemCreateMemory(
      @P("Destination space UUID") String spaceId,
      @P(value = "Text to store; omit when providing filePath", required = false)
          String textContent,
      @P(
              value = "File within the application's enabled upload directory; omit for text",
              required = false)
          String filePath,
      @P(value = "Optional memory metadata", required = false) Map<String, Object> metadata,
      @P(value = "Wait until this memory is indexed; defaults to true", required = false)
          Boolean wait) {
    if ((textContent == null) == (filePath == null)) {
      throw new IllegalArgumentException("Provide exactly one of textContent or filePath");
    }
    var request = JsonMemoryCreationRequest.builder().spaceId(spaceId).metadata(metadata);
    if (textContent != null) {
      request.originalContent(textContent).contentType("text/plain");
    } else {
      if (uploadDirectory == null) {
        throw new IllegalArgumentException(
            "Local file uploads are disabled; provide textContent instead");
      }
      try {
        Path path = uploadDirectory.resolve(filePath).toRealPath();
        if (!path.startsWith(uploadDirectory) || !Files.isRegularFile(path)) {
          throw new IllegalArgumentException("File must be inside the configured upload directory");
        }
        String contentType = Files.probeContentType(path);
        request
            .originalContentBytes(Files.readAllBytes(path))
            .contentType(contentType == null ? "application/octet-stream" : contentType);
      } catch (IOException e) {
        throw new UncheckedIOException("Cannot read file " + filePath, e);
      }
    }
    Memory memory = client.memories.create(request.build());
    return Boolean.FALSE.equals(wait)
        ? memory
        : GoodMemIndexing.waitForMemory(client, memory.memoryId(), indexingTimeout);
  }

  /**
   * Returns compact retrieval results with an explicit partial flag and server diagnostics.
   *
   * @param query natural-language search query
   * @param spaceIds one or more space UUIDs
   * @param maxResults positive result limit, or null for five
   * @param fetchK candidate count at least maxResults, or null for the default
   * @param filter optional native metadata filter applied to every space
   * @param rerankerId optional reranker UUID; reranking requires no LLM
   * @param llmId optional LLM UUID for summarization
   * @return matching text, sources, scores, optional summary and diagnostic statuses
   */
  @Tool(
      "Search GoodMem spaces. Returns text, sources and scores. If partial is true, explain the diagnostics instead of treating results as complete. Optional reranking needs no LLM.")
  public RetrievalResult goodmemRetrieveMemories(
      @P("Natural-language search query") String query,
      @P("Space UUIDs to search") List<String> spaceIds,
      @P(value = "Maximum returned chunks; defaults to five", required = false) Integer maxResults,
      @P(
              value = "Candidates before reranking; defaults to twenty with a reranker",
              required = false)
          Integer fetchK,
      @P(value = "Native GoodMem metadata filter applied to each space", required = false)
          String filter,
      @P(value = "Optional reranker UUID", required = false) String rerankerId,
      @P(value = "Optional LLM UUID for a generated summary", required = false) String llmId) {
    int limit = GoodMemContentRetriever.positive(maxResults == null ? 5 : maxResults, "maxResults");
    int candidates =
        fetchK == null
            ? (rerankerId == null ? limit : Math.max(limit, 20))
            : GoodMemContentRetriever.positive(fetchK, "fetchK");
    if (candidates < limit) {
      throw new IllegalArgumentException("fetchK must be at least maxResults");
    }
    if (spaceIds == null || spaceIds.isEmpty()) {
      throw new IllegalArgumentException("At least one spaceId is required");
    }
    var request =
        RetrieveMemoryRequest.builder()
            .message(query)
            .spaceKeys(
                spaceIds.stream().map(id -> new SpaceKey(SpaceId.from(id), null, filter)).toList())
            .requestedSize(candidates)
            .fetchMemory(true)
            .fetchMemoryContent(false);
    if (rerankerId != null || llmId != null) {
      request.postProcessor(
          ChatPostProcessorConfig.builder()
              .rerankerId(rerankerId)
              .llmId(llmId)
              .maxResults((long) limit)
              .build());
    }
    return RetrievalResults.compact(RetrievalResults.read(client, request.build()), limit);
  }

  /**
   * Gets memory metadata and readable original text. Binary data remains available through the SDK.
   *
   * @param memoryId memory UUID
   * @param includeContent whether to include readable text; null means true
   * @return metadata, readable text and any content notice
   */
  @Tool("Read a stored memory's text and metadata. Binary content is omitted with a notice.")
  public MemoryResult goodmemGetMemory(
      @P("Memory UUID") String memoryId,
      @P(value = "Include readable original text; defaults to true", required = false)
          Boolean includeContent) {
    Memory memory =
        client.memories.get(
            memoryId,
            MemoryGetOptions.builder()
                .includeContent(!Boolean.FALSE.equals(includeContent))
                .build());
    String text = null;
    String notice = null;
    if (!Boolean.FALSE.equals(includeContent)) {
      if (memory.originalContent() == null) {
        notice = "Original content was not available in the response.";
      } else {
        String type =
            memory.contentType() == null
                ? ""
                : memory.contentType().split(";", 2)[0].trim().toLowerCase(Locale.ROOT);
        if (type.startsWith("text/")
            || type.equals("application/json")
            || type.endsWith("+json")
            || type.equals("application/xml")
            || type.endsWith("+xml")
            || type.equals("application/yaml")) {
          try {
            Charset charset = StandardCharsets.UTF_8;
            for (String part : memory.contentType().split(";")) {
              String value = part.trim();
              if (value.toLowerCase(Locale.ROOT).startsWith("charset=")) {
                charset = Charset.forName(value.substring(8).trim().replace("\"", ""));
              }
            }
            text =
                charset.newDecoder().decode(ByteBuffer.wrap(memory.originalContent())).toString();
          } catch (CharacterCodingException | IllegalArgumentException e) {
            notice =
                "Original content could not be decoded as text. Use the SDK to read the bytes.";
          }
        } else {
          notice = "Binary content omitted. Use the SDK to read the original bytes.";
        }
      }
    }
    return new MemoryResult(
        memory.memoryId(),
        memory.spaceId(),
        memory.contentType(),
        memory.processingStatus(),
        memory.metadata(),
        memory.originalContentRef(),
        text,
        notice);
  }

  /**
   * Compact retrieval output for an agent.
   *
   * @param chunks matching text and citation identifiers
   * @param abstractReply optional generated summary
   * @param statuses server diagnostics, including UNKNOWN codes
   * @param partial whether a noninformational diagnostic or missing result data was encountered
   */
  public record RetrievalResult(
      List<RetrievedChunk> chunks,
      String abstractReply,
      List<GoodMemStatus> statuses,
      boolean partial) {}

  /**
   * A retrieved chunk with citation information.
   *
   * @param text matching text
   * @param source stored source reference, when available
   * @param score server relevance score
   * @param memoryId memory identifier
   * @param chunkId chunk identifier
   * @param spaceId containing space, when available
   */
  public record RetrievedChunk(
      String text,
      String source,
      Double score,
      MemoryId memoryId,
      ChunkId chunkId,
      SpaceId spaceId) {}

  /**
   * Agent-readable memory output without base64 content.
   *
   * @param memoryId memory identifier
   * @param spaceId containing space
   * @param contentType original media type
   * @param processingStatus indexing status
   * @param metadata stored user metadata
   * @param originalContentRef stored source reference
   * @param text decoded original text, when requested and available
   * @param contentNotice explanation when requested text is unavailable
   */
  public record MemoryResult(
      MemoryId memoryId,
      SpaceId spaceId,
      String contentType,
      MemoryProcessingStatus processingStatus,
      Map<String, Object> metadata,
      String originalContentRef,
      String text,
      String contentNotice) {}

  /**
   * Deletes the requested memory through the SDK.
   *
   * @param memoryId memory UUID
   */
  @Tool("Permanently delete a GoodMem memory and its chunks.")
  public void goodmemDeleteMemory(@P("Memory UUID") String memoryId) {
    client.memories.delete(memoryId);
  }

  /**
   * Returns available embedder models.
   *
   * @return embedder configurations from the SDK
   */
  @Tool("List the available GoodMem embedder models.")
  public List<EmbedderResponse> goodmemListEmbedders() {
    return client.embedders.list();
  }

  /**
   * Lists spaces across SDK pages, up to the requested maximum.
   *
   * @param nameFilter optional name filter
   * @param maxItems positive maximum number of spaces, or null for 100
   * @return matching spaces in server order
   */
  @Tool(
      "List GoodMem spaces, optionally filtered by name. Fetches across pages up to maxItems, which defaults to 100.")
  public List<Space> goodmemListSpaces(
      @P(value = "Optional name filter", required = false) String nameFilter,
      @P(value = "Maximum spaces to return; defaults to 100", required = false) Integer maxItems) {
    return collect(
        client.spaces.list(SpaceListOptions.builder().nameFilter(nameFilter).build()), maxItems);
  }

  /**
   * Gets a space by ID.
   *
   * @param spaceId space UUID
   * @return the requested space
   */
  @Tool("Get a GoodMem space's name, labels and embedder configuration.")
  public Space goodmemGetSpace(@P("Space UUID") String spaceId) {
    return client.spaces.get(spaceId);
  }

  /**
   * Updates supported mutable fields. Label replacement and merging are mutually exclusive.
   *
   * @param spaceId space UUID
   * @param name new name, or null to keep the current name
   * @param replaceLabels replacement labels, or null
   * @param mergeLabels labels to merge, or null
   * @return the updated space
   */
  @Tool("Update a GoodMem space's name or labels. Choose label replacement or merging, not both.")
  public Space goodmemUpdateSpace(
      @P("Space UUID") String spaceId,
      @P(value = "New name", required = false) String name,
      @P(value = "Labels to replace existing labels", required = false)
          Map<String, String> replaceLabels,
      @P(value = "Labels to merge with existing labels", required = false)
          Map<String, String> mergeLabels) {
    return client.spaces.update(
        spaceId,
        UpdateSpaceRequest.builder()
            .name(name)
            .replaceLabels(replaceLabels)
            .mergeLabels(mergeLabels)
            .build());
  }

  /**
   * Deletes a space and all its memories through the SDK.
   *
   * @param spaceId space UUID
   */
  @Tool("Permanently delete a GoodMem space and all its memories.")
  public void goodmemDeleteSpace(@P("Space UUID") String spaceId) {
    client.spaces.delete(spaceId);
  }

  /**
   * Lists memory metadata across SDK pages with optional processing and metadata filters.
   *
   * @param spaceId space UUID
   * @param maxItems positive maximum number of memories, or null for 100
   * @param statusFilter optional processing status
   * @param filter optional native metadata filter
   * @return matching memories, without original content, in server order
   */
  @Tool("List memory metadata in a space, fetching across pages up to maxItems (100 by default).")
  public List<Memory> goodmemListMemories(
      @P("Space UUID") String spaceId,
      @P(value = "Maximum memories to return; defaults to 100", required = false) Integer maxItems,
      @P(value = "Optional processing status", required = false)
          MemoriesListStatusFilter statusFilter,
      @P(value = "Optional native metadata filter", required = false) String filter) {
    return collect(
        client.memories.list(
            spaceId,
            MemoryListOptions.builder()
                .statusFilter(statusFilter)
                .filter(filter)
                .includeContent(false)
                .build()),
        maxItems);
  }

  private static <T> List<T> collect(Page<T> page, Integer maxItems) {
    int limit = GoodMemContentRetriever.positive(maxItems == null ? 100 : maxItems, "maxItems");
    List<T> items = new ArrayList<>();
    page.iterator(limit).forEachRemaining(items::add);
    return List.copyOf(items);
  }
}
