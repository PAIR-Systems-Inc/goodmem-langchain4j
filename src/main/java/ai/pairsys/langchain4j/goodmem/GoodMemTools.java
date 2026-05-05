package ai.pairsys.langchain4j.goodmem;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;

import java.lang.reflect.Type;
import java.time.Duration;
import java.util.List;
import java.util.Map;

import static dev.langchain4j.internal.ValidationUtils.ensureNotBlank;

/**
 * GoodMem tools for LangChain4j agents.
 * <p>
 * Provides {@code @Tool}-annotated methods for creating spaces, storing memories,
 * performing semantic retrieval, and managing memories in GoodMem. Each tool method
 * returns a JSON string that the LLM can parse and act upon.
 * <p>
 * Usage:
 * <pre>{@code
 * GoodMemTools tools = GoodMemTools.builder()
 *         .baseUrl("https://localhost:8080")
 *         .apiKey("your-api-key")
 *         .build();
 *
 * // Register with an AI service
 * Assistant assistant = AiServices.builder(Assistant.class)
 *         .chatLanguageModel(model)
 *         .tools(tools)
 *         .build();
 * }</pre>
 */
public class GoodMemTools {

    private static final Gson GSON = new Gson();

    private final GoodMemClient client;

    private GoodMemTools(GoodMemClient client) {
        this.client = client;
    }

    /**
     * Create a new GoodMem space or reuse an existing one.
     * A space is a logical container for organizing related memories,
     * configured with an embedder for vector search.
     *
     * @param name              a unique name for the space
     * @param embedderId        the ID of the embedder model that converts text into vector representations
     * @param chunkingStrategy  the chunking strategy: 'recursive', 'sentence', or 'none'
     * @param chunkSize         maximum chunk size in characters (for recursive/sentence)
     * @param chunkOverlap      overlap between consecutive chunks in characters
     * @return JSON string with the operation result including spaceId
     */
    @Tool("Create a new GoodMem space or reuse an existing one. "
            + "A space is a logical container for organizing related memories, "
            + "configured with an embedder for vector search.")
    public String goodmemCreateSpace(
            @P("A unique name for the space") String name,
            @P("The ID of the embedder model that converts text into vector representations for similarity search") String embedderId,
            @P(value = "The chunking strategy for text processing: 'recursive', 'sentence', or 'none'", required = false) String chunkingStrategy,
            @P(value = "Maximum chunk size in characters (for recursive/sentence strategies)", required = false) Integer chunkSize,
            @P(value = "Overlap between consecutive chunks in characters", required = false) Integer chunkOverlap) {
        try {
            JsonObject result = client.createSpace(
                    name,
                    embedderId,
                    chunkingStrategy != null ? chunkingStrategy : "recursive",
                    chunkSize != null ? chunkSize : 512,
                    chunkOverlap != null ? chunkOverlap : 50);
            return GSON.toJson(result);
        } catch (Exception e) {
            return errorJson(e);
        }
    }

    /**
     * Store a document as a new memory in a GoodMem space.
     * Accepts a local file path or plain text. The memory is chunked and embedded asynchronously.
     *
     * @param spaceId     the UUID of the space to store the memory in
     * @param textContent plain text content to store as memory
     * @param filePath    local file path to upload as memory (PDF, DOCX, image, etc.)
     * @param metadataJson optional JSON string of key-value metadata to attach to the memory
     * @return JSON string with the operation result including memoryId
     */
    @Tool("Store a document as a new memory in a GoodMem space. "
            + "Accepts a local file path or plain text. "
            + "The memory is chunked and embedded asynchronously.")
    public String goodmemCreateMemory(
            @P("The UUID of the space to store the memory in") String spaceId,
            @P(value = "Plain text content to store as memory. If both filePath and textContent are provided, the file takes priority.", required = false) String textContent,
            @P(value = "Local file path to upload as memory (PDF, DOCX, image, etc.). Content type is auto-detected.", required = false) String filePath,
            @P(value = "Optional JSON string of key-value metadata to attach to the memory, e.g. '{\"source\":\"email\",\"author\":\"John\"}'", required = false) String metadataJson) {
        try {
            Map<String, Object> metadata = null;
            if (metadataJson != null && !metadataJson.isEmpty()) {
                Type mapType = new com.google.gson.reflect.TypeToken<Map<String, Object>>() {}.getType();
                metadata = GSON.fromJson(metadataJson, mapType);
            }
            JsonObject result = client.createMemory(spaceId, textContent, filePath, metadata);
            return GSON.toJson(result);
        } catch (Exception e) {
            return errorJson(e);
        }
    }

    /**
     * Perform similarity-based semantic retrieval across one or more GoodMem spaces.
     * Returns matching chunks ranked by relevance.
     * <p>
     * When any of {@code rerankerId}, {@code llmId}, {@code relevanceThreshold},
     * {@code llmTemperature}, or {@code chronologicalResort} is provided, the
     * server appends a {@code ChatPostProcessor} stage that reranks, filters,
     * re-sorts, and/or generates an LLM summary ({@code abstractReply}).
     *
     * @param query                    a natural language query for semantic search
     * @param spaceIds                 one or more space UUIDs separated by commas
     * @param maxResults               maximum number of matching chunks to return
     * @param includeMemoryDefinition  fetch the full memory metadata alongside matched chunks
     * @param waitForIndexing          retry for up to 10 seconds when no results are found
     * @param rerankerId               optional UUID of a reranker model to refine ordering
     * @param llmId                    optional UUID of an LLM that generates a contextual abstractReply
     * @param relevanceThreshold       optional minimum relevance score (0-1) for inclusion
     * @param llmTemperature           optional LLM temperature for generation (0-2)
     * @param chronologicalResort      optional flag to reorder final results by creation time
     * @return JSON string with the retrieval results
     */
    @Tool("Perform similarity-based semantic retrieval across one or more GoodMem spaces. "
            + "Returns matching chunks ranked by relevance. Optionally rerank via a reranker model, "
            + "filter by relevance score, generate an LLM summary (abstractReply), and resort by time.")
    public String goodmemRetrieveMemories(
            @P("A natural language query used to find semantically similar memory chunks") String query,
            @P("One or more space UUIDs to search across, separated by commas (e.g., 'id1,id2')") String spaceIds,
            @P(value = "Maximum number of matching chunks to return", required = false) Integer maxResults,
            @P(value = "Fetch the full memory metadata alongside the matched chunks", required = false) Boolean includeMemoryDefinition,
            @P(value = "Retry for up to 10 seconds when no results are found (use when memories were just added)", required = false) Boolean waitForIndexing,
            @P(value = "UUID of a reranker model to refine the order of retrieved chunks", required = false) String rerankerId,
            @P(value = "UUID of an LLM that produces a contextual summary (abstractReply) over the retrieved chunks", required = false) String llmId,
            @P(value = "Minimum relevance score (0-1) below which results are dropped (only with a post-processor)", required = false) Double relevanceThreshold,
            @P(value = "Creativity setting for LLM generation (0-2). Only used when llmId is also provided.", required = false) Double llmTemperature,
            @P(value = "Reorder final results by creation time after reranking and thresholding", required = false) Boolean chronologicalResort) {
        try {
            JsonObject result = client.retrieveMemories(
                    query,
                    spaceIds,
                    maxResults != null ? maxResults : 5,
                    includeMemoryDefinition != null ? includeMemoryDefinition : true,
                    waitForIndexing != null ? waitForIndexing : true,
                    rerankerId,
                    llmId,
                    relevanceThreshold,
                    llmTemperature,
                    chronologicalResort);
            return GSON.toJson(result);
        } catch (Exception e) {
            return errorJson(e);
        }
    }

    /**
     * Fetch a specific GoodMem memory by its ID, including metadata,
     * processing status, and optionally the original content.
     *
     * @param memoryId       the UUID of the memory to fetch
     * @param includeContent fetch the original document content in addition to metadata
     * @return JSON string with the memory data
     */
    @Tool("Fetch a specific GoodMem memory by its ID, including metadata, "
            + "processing status, and optionally the original content.")
    public String goodmemGetMemory(
            @P("The UUID of the memory to fetch") String memoryId,
            @P(value = "Fetch the original document content of the memory in addition to its metadata", required = false) Boolean includeContent) {
        try {
            JsonObject result = client.getMemory(
                    memoryId,
                    includeContent != null ? includeContent : true);
            return GSON.toJson(result);
        } catch (Exception e) {
            return errorJson(e);
        }
    }

    /**
     * Permanently delete a GoodMem memory and its associated chunks and vector embeddings.
     *
     * @param memoryId the UUID of the memory to delete
     * @return JSON string with the deletion result
     */
    @Tool("Permanently delete a GoodMem memory and its associated chunks and vector embeddings.")
    public String goodmemDeleteMemory(
            @P("The UUID of the memory to delete") String memoryId) {
        try {
            JsonObject result = client.deleteMemory(memoryId);
            return GSON.toJson(result);
        } catch (Exception e) {
            return errorJson(e);
        }
    }

    /**
     * List all available GoodMem embedder models.
     * Use the returned embedder ID when creating a new space.
     *
     * @return JSON string with the list of embedders
     */
    @Tool("List all available GoodMem embedder models. "
            + "Use the returned embedder ID when creating a new space.")
    public String goodmemListEmbedders() {
        try {
            List<JsonObject> embedders = client.listEmbedders();
            JsonObject result = new JsonObject();
            result.addProperty("success", true);
            result.add("embedders", GSON.toJsonTree(embedders));
            result.addProperty("totalEmbedders", embedders.size());
            return GSON.toJson(result);
        } catch (Exception e) {
            return errorJson(e);
        }
    }

    /**
     * List all GoodMem spaces. Returns each space with its ID, name,
     * embedder configuration, and access settings.
     *
     * @return JSON string with the list of spaces
     */
    @Tool("List all GoodMem spaces. Returns each space with its ID, name, "
            + "embedder configuration, and access settings.")
    public String goodmemListSpaces() {
        try {
            List<JsonObject> spaces = client.listSpaces();
            JsonObject result = new JsonObject();
            result.addProperty("success", true);
            result.add("spaces", GSON.toJsonTree(spaces));
            result.addProperty("totalSpaces", spaces.size());
            return GSON.toJson(result);
        } catch (Exception e) {
            return errorJson(e);
        }
    }

    /**
     * Fetch a specific GoodMem space by its ID, including name, labels,
     * embedder configuration, and metadata.
     *
     * @param spaceId the UUID of the space to fetch
     * @return JSON string with the space data
     */
    @Tool("Fetch a specific GoodMem space by its ID, including name, labels, "
            + "embedder configuration, and metadata.")
    public String goodmemGetSpace(
            @P("The UUID of the space to fetch") String spaceId) {
        try {
            JsonObject result = client.getSpace(spaceId);
            return GSON.toJson(result);
        } catch (Exception e) {
            return errorJson(e);
        }
    }

    /**
     * Update mutable fields on an existing GoodMem space.
     * Only {@code name}, {@code publicRead}, and labels are mutable;
     * embedders and chunking config are immutable after creation.
     * {@code replaceLabels} and {@code mergeLabels} are mutually exclusive.
     *
     * @param spaceId            the UUID of the space to update
     * @param name               new name for the space, or null to keep current
     * @param publicRead         whether the space should be publicly readable, or null
     * @param replaceLabelsJson  JSON object string of labels to replace all existing labels
     * @param mergeLabelsJson    JSON object string of labels to merge with existing labels
     * @return JSON string with the updated space
     */
    @Tool("Update mutable fields on an existing GoodMem space. Only name, publicRead, and labels "
            + "can be modified. replaceLabels and mergeLabels are mutually exclusive.")
    public String goodmemUpdateSpace(
            @P("The UUID of the space to update") String spaceId,
            @P(value = "New name for the space (omit to leave unchanged)", required = false) String name,
            @P(value = "Whether the space should be publicly readable (omit to leave unchanged)", required = false) Boolean publicRead,
            @P(value = "JSON object string of labels to fully replace existing labels (e.g., '{\"team\":\"nlp\"}'). Mutually exclusive with mergeLabels.", required = false) String replaceLabelsJson,
            @P(value = "JSON object string of labels to merge into existing labels. Mutually exclusive with replaceLabelsJson.", required = false) String mergeLabelsJson) {
        try {
            Map<String, String> replaceLabels = parseLabelsJson(replaceLabelsJson);
            Map<String, String> mergeLabels = parseLabelsJson(mergeLabelsJson);
            JsonObject result = client.updateSpace(spaceId, name, publicRead, replaceLabels, mergeLabels);
            return GSON.toJson(result);
        } catch (Exception e) {
            return errorJson(e);
        }
    }

    /**
     * Permanently delete a GoodMem space and all of its memories.
     *
     * @param spaceId the UUID of the space to delete
     * @return JSON string with the deletion result
     */
    @Tool("Permanently delete a GoodMem space and all of its memories. "
            + "This operation cannot be undone.")
    public String goodmemDeleteSpace(
            @P("The UUID of the space to delete") String spaceId) {
        try {
            JsonObject result = client.deleteSpace(spaceId);
            return GSON.toJson(result);
        } catch (Exception e) {
            return errorJson(e);
        }
    }

    /**
     * List memories within a GoodMem space, with optional pagination, status
     * filtering, and metadata filter expressions.
     *
     * @param spaceId           the UUID of the space whose memories to list
     * @param maxResults        page size; server clamps to a sensible range
     * @param nextToken         opaque pagination token from a prior list_memories response
     * @param statusFilter      optional processing-status filter (PENDING, PROCESSING, COMPLETED, FAILED)
     * @param includeContent    whether to include the original content in the response
     * @param filterExpression  optional GoodMem filter expression
     * @return JSON string with the memories list and an optional nextToken
     */
    @Tool("List memories in a GoodMem space, with optional pagination, "
            + "status filtering, and metadata filter expressions.")
    public String goodmemListMemories(
            @P("The UUID of the space whose memories to list") String spaceId,
            @P(value = "Maximum memories to return per page (server clamps to its allowed range)", required = false) Integer maxResults,
            @P(value = "Opaque pagination token from a prior list_memories response", required = false) String nextToken,
            @P(value = "Filter by processing status: PENDING, PROCESSING, COMPLETED, or FAILED", required = false) String statusFilter,
            @P(value = "Whether to include the original content for each memory", required = false) Boolean includeContent,
            @P(value = "Metadata filter expression using GoodMem filter syntax (e.g., \"val('$.source') = 'email'\")", required = false) String filterExpression) {
        try {
            JsonObject result = client.listMemories(
                    spaceId,
                    maxResults,
                    nextToken,
                    statusFilter,
                    includeContent != null && includeContent,
                    filterExpression);
            return GSON.toJson(result);
        } catch (Exception e) {
            return errorJson(e);
        }
    }

    private static Map<String, String> parseLabelsJson(String json) {
        if (json == null || json.isEmpty()) {
            return null;
        }
        try {
            JsonObject obj = GSON.fromJson(json, JsonObject.class);
            if (obj == null) {
                return null;
            }
            java.util.LinkedHashMap<String, String> labels = new java.util.LinkedHashMap<>();
            for (Map.Entry<String, com.google.gson.JsonElement> entry : obj.entrySet()) {
                if (!entry.getValue().isJsonNull()) {
                    labels.put(entry.getKey(), entry.getValue().getAsString());
                }
            }
            return labels;
        } catch (Exception e) {
            throw new IllegalArgumentException(
                    "Labels must be a JSON object of string-to-string pairs, got: " + json, e);
        }
    }

    private static String errorJson(Exception e) {
        JsonObject error = new JsonObject();
        error.addProperty("success", false);
        error.addProperty("error", e.getMessage());
        return GSON.toJson(error);
    }

    /**
     * Creates a new {@link Builder} for constructing a {@link GoodMemTools} instance.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for {@link GoodMemTools}.
     */
    public static final class Builder {

        private String baseUrl;
        private String apiKey;
        private Duration timeout = Duration.ofSeconds(30);
        private boolean verifySsl = true;

        private Builder() {
        }

        /**
         * Sets the GoodMem API base URL.
         *
         * @param baseUrl the base URL (e.g., "https://localhost:8080")
         * @return this builder
         */
        public Builder baseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
            return this;
        }

        /**
         * Sets the GoodMem API key.
         *
         * @param apiKey the API key for authentication
         * @return this builder
         */
        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        /**
         * Sets the request timeout. Defaults to 30 seconds.
         *
         * @param timeout the timeout duration
         * @return this builder
         */
        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        /**
         * Sets whether to verify SSL certificates. Defaults to true.
         *
         * @param verifySsl true to verify SSL certificates
         * @return this builder
         */
        public Builder verifySsl(boolean verifySsl) {
            this.verifySsl = verifySsl;
            return this;
        }

        /**
         * Builds a new {@link GoodMemTools} instance.
         *
         * @return the configured tools instance
         */
        public GoodMemTools build() {
            ensureNotBlank(baseUrl, "baseUrl");
            ensureNotBlank(apiKey, "apiKey");
            GoodMemClient client = new GoodMemClient(baseUrl, apiKey, timeout, verifySsl);
            return new GoodMemTools(client);
        }
    }
}
