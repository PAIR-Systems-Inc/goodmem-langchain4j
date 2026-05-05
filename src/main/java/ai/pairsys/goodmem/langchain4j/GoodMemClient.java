package ai.pairsys.goodmem.langchain4j;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.reflect.Type;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;

import static dev.langchain4j.internal.ValidationUtils.ensureNotBlank;
import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;

/**
 * Low-level HTTP client for communicating with the GoodMem API.
 * <p>
 * Handles authentication, URL normalization, and common request patterns
 * used by all GoodMem tools.
 */
public class GoodMemClient {

    private static final Logger log = LoggerFactory.getLogger(GoodMemClient.class);
    private static final Gson GSON = new Gson();
    private static final int DEFAULT_CHUNK_SIZE = 512;
    private static final int DEFAULT_CHUNK_OVERLAP = 50;
    private static final String CHAT_POSTPROCESSOR =
            "com.goodmem.retrieval.postprocess.ChatPostProcessorFactory";

    private final String baseUrl;
    private final String apiKey;
    private final Duration timeout;
    private final boolean verifySsl;
    private final HttpClient httpClient;

    /**
     * Creates a new GoodMemClient.
     *
     * @param baseUrl   the base URL of the GoodMem API server
     * @param apiKey    the API key for authentication via X-API-Key header
     * @param timeout   request timeout
     * @param verifySsl whether to verify SSL certificates
     */
    public GoodMemClient(String baseUrl, String apiKey, Duration timeout, boolean verifySsl) {
        this.baseUrl = ensureNotBlank(baseUrl, "baseUrl").replaceAll("/+$", "");
        this.apiKey = ensureNotBlank(apiKey, "apiKey");
        this.timeout = ensureNotNull(timeout, "timeout");
        this.verifySsl = verifySsl;
        this.httpClient = buildHttpClient();
    }

    /**
     * Returns the configured base URL of the GoodMem API server.
     *
     * @return the base URL (without trailing slash)
     */
    public String baseUrl() {
        return baseUrl;
    }

    // -- Space operations --

    /**
     * Create a new space or return an existing one with the same name.
     * <p>
     * First lists existing spaces to check for a name match. If found,
     * returns the existing space info. Otherwise creates a new space.
     *
     * @param name              human-readable name of the space
     * @param embedderId        UUID of the embedder model to use
     * @param chunkingStrategy  chunking strategy id (e.g. "fixed", "none")
     * @param chunkSize         maximum chunk size in tokens (ignored when strategy is "none")
     * @param chunkOverlap      number of overlapping tokens between consecutive chunks
     * @return result envelope containing {@code spaceId}, {@code name}, {@code embedderId},
     *         and a {@code reused} boolean indicating whether an existing space was returned
     */
    public JsonObject createSpace(String name, String embedderId,
                                  String chunkingStrategy, int chunkSize, int chunkOverlap) {
        // Check for existing space with the same name
        try {
            List<JsonObject> spaces = listSpaces();
            for (JsonObject space : spaces) {
                if (space.has("name") && name.equals(space.get("name").getAsString())) {
                    String actualEmbedderId = embedderId;
                    if (space.has("spaceEmbedders")) {
                        JsonArray embedders = space.getAsJsonArray("spaceEmbedders");
                        if (!embedders.isEmpty()) {
                            JsonObject first = embedders.get(0).getAsJsonObject();
                            if (first.has("embedderId")) {
                                actualEmbedderId = first.get("embedderId").getAsString();
                            }
                        }
                    }
                    JsonObject result = new JsonObject();
                    result.addProperty("success", true);
                    result.addProperty("spaceId", space.get("spaceId").getAsString());
                    result.addProperty("name", space.get("name").getAsString());
                    result.addProperty("embedderId", actualEmbedderId);
                    result.addProperty("message", "Space already exists, reusing existing space");
                    result.addProperty("reused", true);
                    return result;
                }
            }
        } catch (Exception e) {
            log.debug("Failed to list spaces for deduplication check, proceeding to create: {}", e.getMessage());
        }

        JsonObject chunkingConfig = new JsonObject();
        if ("none".equals(chunkingStrategy)) {
            chunkingConfig.add("none", new JsonObject());
        } else {
            JsonObject strategyConfig = new JsonObject();
            strategyConfig.addProperty("chunkSize", chunkSize);
            strategyConfig.addProperty("chunkOverlap", chunkOverlap);
            chunkingConfig.add(chunkingStrategy, strategyConfig);
        }

        JsonObject embedderRef = new JsonObject();
        embedderRef.addProperty("embedderId", embedderId);
        JsonArray spaceEmbedders = new JsonArray();
        spaceEmbedders.add(embedderRef);

        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("name", name);
        requestBody.add("spaceEmbedders", spaceEmbedders);
        requestBody.add("defaultChunkingConfig", chunkingConfig);

        JsonObject body = post("/v1/spaces", requestBody);

        JsonObject result = new JsonObject();
        result.addProperty("success", true);
        result.addProperty("spaceId", body.get("spaceId").getAsString());
        result.addProperty("name", body.get("name").getAsString());
        result.addProperty("embedderId", embedderId);
        result.addProperty("message", "Space created successfully");
        result.addProperty("reused", false);
        return result;
    }

    /**
     * List all spaces.
     *
     * @return all spaces visible to the authenticated user
     */
    public List<JsonObject> listSpaces() {
        JsonElement body = getJson("/v1/spaces");
        return parseListResponse(body, "spaces");
    }

    /**
     * Fetch a space by ID.
     *
     * @param spaceId UUID of the space to fetch
     * @return result envelope containing the {@code space} object
     */
    public JsonObject getSpace(String spaceId) {
        JsonElement body = getJson("/v1/spaces/" + spaceId);
        JsonObject result = new JsonObject();
        result.addProperty("success", true);
        result.add("space", body);
        return result;
    }

    /**
     * Update mutable fields on a space.
     * <p>
     * Only {@code name}, {@code publicRead}, and labels are mutable.
     * Pass {@code null} for any field that should be left unchanged.
     * {@code replaceLabels} and {@code mergeLabels} are mutually exclusive.
     *
     * @param spaceId       UUID of the space to update
     * @param name          new name, or {@code null} to leave unchanged
     * @param publicRead    new public-read flag, or {@code null} to leave unchanged
     * @param replaceLabels map of labels that fully replaces the existing labels, or {@code null}
     * @param mergeLabels   map of labels merged into the existing labels, or {@code null}
     * @return result envelope containing the updated {@code space} object
     */
    public JsonObject updateSpace(String spaceId, String name, Boolean publicRead,
                                  Map<String, String> replaceLabels,
                                  Map<String, String> mergeLabels) {
        JsonObject requestBody = new JsonObject();
        if (name != null) {
            requestBody.addProperty("name", name);
        }
        if (publicRead != null) {
            requestBody.addProperty("publicRead", publicRead);
        }
        if (replaceLabels != null) {
            requestBody.add("replaceLabels", GSON.toJsonTree(replaceLabels));
        }
        if (mergeLabels != null) {
            requestBody.add("mergeLabels", GSON.toJsonTree(mergeLabels));
        }

        JsonObject body = put("/v1/spaces/" + spaceId, requestBody);

        JsonObject result = new JsonObject();
        result.addProperty("success", true);
        result.add("space", body);
        result.addProperty("message", "Space updated successfully");
        return result;
    }

    /**
     * Delete a space and all its memories by ID.
     *
     * @param spaceId UUID of the space to delete
     * @return result envelope confirming the deletion
     */
    public JsonObject deleteSpace(String spaceId) {
        delete("/v1/spaces/" + spaceId);

        JsonObject result = new JsonObject();
        result.addProperty("success", true);
        result.addProperty("spaceId", spaceId);
        result.addProperty("message", "Space deleted successfully");
        return result;
    }

    // -- Memory operations --

    /**
     * Create a new memory in a space from text or a file.
     * <p>
     * If both filePath and textContent are provided, the file takes priority.
     * The file content type is auto-detected from its extension.
     *
     * @param spaceId     UUID of the space the memory belongs to
     * @param textContent raw text content (used when {@code filePath} is null or empty)
     * @param filePath    local path to a file to upload (takes priority over text content)
     * @param metadata    optional key/value metadata attached to the memory, or {@code null}
     * @return result envelope containing {@code memoryId}, {@code spaceId}, processing
     *         {@code status}, and the resolved {@code contentType}
     */
    public JsonObject createMemory(String spaceId, String textContent, String filePath,
                                   Map<String, Object> metadata) {
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("spaceId", spaceId);

        if (filePath != null && !filePath.isEmpty()) {
            Path path = Path.of(filePath);
            String mimeType = null;
            try {
                mimeType = Files.probeContentType(path);
            } catch (IOException e) {
                log.debug("Could not probe content type for {}: {}", filePath, e.getMessage());
            }
            if (mimeType == null) {
                // Fallback detection based on extension
                String name = path.getFileName().toString().toLowerCase();
                if (name.endsWith(".pdf")) {
                    mimeType = "application/pdf";
                } else if (name.endsWith(".docx")) {
                    mimeType = "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
                } else if (name.endsWith(".txt")) {
                    mimeType = "text/plain";
                } else {
                    mimeType = "application/octet-stream";
                }
            }

            try {
                byte[] fileBytes = Files.readAllBytes(path);
                requestBody.addProperty("contentType", mimeType);

                if (mimeType.startsWith("text/")) {
                    requestBody.addProperty("originalContent", new String(fileBytes, StandardCharsets.UTF_8));
                } else {
                    requestBody.addProperty("originalContentB64",
                            Base64.getEncoder().encodeToString(fileBytes));
                }
            } catch (IOException e) {
                throw new GoodMemException("Failed to read file: " + filePath, e);
            }
        } else if (textContent != null && !textContent.isEmpty()) {
            requestBody.addProperty("contentType", "text/plain");
            requestBody.addProperty("originalContent", textContent);
        } else {
            throw new IllegalArgumentException(
                    "No content provided. Provide either textContent or filePath.");
        }

        if (metadata != null && !metadata.isEmpty()) {
            requestBody.add("metadata", GSON.toJsonTree(metadata));
        }

        JsonObject body = post("/v1/memories", requestBody);

        JsonObject result = new JsonObject();
        result.addProperty("success", true);
        result.addProperty("memoryId", body.get("memoryId").getAsString());
        result.addProperty("spaceId", body.get("spaceId").getAsString());
        result.addProperty("status",
                body.has("processingStatus") ? body.get("processingStatus").getAsString() : "PENDING");
        result.addProperty("contentType", requestBody.get("contentType").getAsString());
        result.addProperty("message", "Memory created successfully");
        return result;
    }

    /**
     * Retrieve memories via semantic similarity search.
     * <p>
     * Supports polling for up to 10 seconds when waitForIndexing is enabled
     * and no results are found initially.
     * <p>
     * When any of {@code rerankerId}, {@code llmId}, {@code relevanceThreshold},
     * {@code llmTemperature}, or {@code chronologicalResort} is provided, a
     * {@code ChatPostProcessor} stage is appended that reranks, filters,
     * re-sorts, and/or generates an LLM summary ({@code abstractReply}).
     *
     * @param query                    natural-language query
     * @param spaceIds                 one or more space UUIDs to search across, comma-separated
     * @param maxResults               maximum number of matching chunks to return
     * @param includeMemoryDefinition  whether to include full memory metadata alongside chunks
     * @param waitForIndexing          whether to poll for up to 10 seconds when results are empty
     * @param rerankerId               UUID of an optional reranker model, or {@code null}
     * @param llmId                    UUID of an optional LLM that produces an {@code abstractReply}, or {@code null}
     * @param relevanceThreshold       minimum relevance score (0-1) to keep, or {@code null}
     * @param llmTemperature           creativity setting (0-2) for LLM generation, or {@code null}
     * @param chronologicalResort      whether to reorder final results by creation time, or {@code null}
     * @return result envelope containing {@code resultSetId}, {@code results}, {@code memories},
     *         {@code totalResults}, {@code query}, and optionally {@code abstractReply}
     */
    public JsonObject retrieveMemories(String query, String spaceIds, int maxResults,
                                       boolean includeMemoryDefinition, boolean waitForIndexing,
                                       String rerankerId, String llmId,
                                       Double relevanceThreshold, Double llmTemperature,
                                       Boolean chronologicalResort) {
        String[] ids = spaceIds.split(",");
        JsonArray spaceKeys = new JsonArray();
        for (String id : ids) {
            String trimmed = id.trim();
            if (!trimmed.isEmpty()) {
                JsonObject key = new JsonObject();
                key.addProperty("spaceId", trimmed);
                spaceKeys.add(key);
            }
        }
        if (spaceKeys.isEmpty()) {
            throw new IllegalArgumentException("At least one valid Space ID is required.");
        }

        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("message", query);
        requestBody.add("spaceKeys", spaceKeys);
        requestBody.addProperty("requestedSize", maxResults);
        requestBody.addProperty("fetchMemory", includeMemoryDefinition);

        JsonObject postConfig = new JsonObject();
        if (rerankerId != null && !rerankerId.isEmpty()) {
            postConfig.addProperty("reranker_id", rerankerId);
        }
        if (llmId != null && !llmId.isEmpty()) {
            postConfig.addProperty("llm_id", llmId);
        }
        if (relevanceThreshold != null) {
            postConfig.addProperty("relevance_threshold", relevanceThreshold);
        }
        if (llmTemperature != null) {
            postConfig.addProperty("llm_temp", llmTemperature);
        }
        if (chronologicalResort != null) {
            postConfig.addProperty("chronological_resort", chronologicalResort);
        }
        if (postConfig.size() > 0) {
            if (!postConfig.has("max_results")) {
                postConfig.addProperty("max_results", maxResults);
            }
            JsonObject postProcessor = new JsonObject();
            postProcessor.addProperty("name", CHAT_POSTPROCESSOR);
            postProcessor.add("config", postConfig);
            requestBody.add("postProcessor", postProcessor);
        }

        long maxWaitMs = 10_000;
        long pollIntervalMs = 2_000;
        long start = System.currentTimeMillis();
        JsonObject lastResult = null;

        while (true) {
            String responseText = postNdjson("/v1/memories:retrieve", requestBody);

            List<JsonObject> results = new ArrayList<>();
            List<JsonObject> memories = new ArrayList<>();
            String resultSetId = "";
            JsonObject abstractReply = null;

            for (String line : responseText.split("\n")) {
                String jsonStr = line.trim();
                if (jsonStr.isEmpty()) continue;
                if (jsonStr.startsWith("data:")) {
                    jsonStr = jsonStr.substring(5).trim();
                }
                if (jsonStr.startsWith("event:") || jsonStr.isEmpty()) continue;

                try {
                    JsonObject item = GSON.fromJson(jsonStr, JsonObject.class);

                    if (item.has("resultSetBoundary")) {
                        JsonObject boundary = item.getAsJsonObject("resultSetBoundary");
                        if (boundary.has("resultSetId")) {
                            resultSetId = boundary.get("resultSetId").getAsString();
                        }
                    } else if (item.has("memoryDefinition")) {
                        memories.add(item.getAsJsonObject("memoryDefinition"));
                    } else if (item.has("abstractReply")) {
                        abstractReply = item.getAsJsonObject("abstractReply");
                    } else if (item.has("retrievedItem")) {
                        JsonObject ri = item.getAsJsonObject("retrievedItem");
                        JsonObject chunkData = ri.has("chunk") ? ri.getAsJsonObject("chunk") : new JsonObject();
                        JsonObject chunk = chunkData.has("chunk") ? chunkData.getAsJsonObject("chunk") : new JsonObject();

                        JsonObject parsed = new JsonObject();
                        parsed.addProperty("chunkId",
                                chunk.has("chunkId") ? chunk.get("chunkId").getAsString() : "");
                        parsed.addProperty("chunkText",
                                chunk.has("chunkText") ? chunk.get("chunkText").getAsString() : "");
                        parsed.addProperty("memoryId",
                                chunk.has("memoryId") ? chunk.get("memoryId").getAsString() : "");
                        if (chunkData.has("relevanceScore")) {
                            parsed.addProperty("relevanceScore", chunkData.get("relevanceScore").getAsDouble());
                        }
                        if (chunkData.has("memoryIndex")) {
                            parsed.addProperty("memoryIndex", chunkData.get("memoryIndex").getAsInt());
                        }
                        results.add(parsed);
                    }
                } catch (Exception e) {
                    log.debug("Skipping unparseable NDJSON line: {}", jsonStr);
                }
            }

            lastResult = new JsonObject();
            lastResult.addProperty("success", true);
            lastResult.addProperty("resultSetId", resultSetId);
            lastResult.add("results", GSON.toJsonTree(results));
            lastResult.add("memories", GSON.toJsonTree(memories));
            lastResult.addProperty("totalResults", results.size());
            lastResult.addProperty("query", query);
            if (abstractReply != null) {
                lastResult.add("abstractReply", abstractReply);
            }

            if (!results.isEmpty() || !waitForIndexing) {
                return lastResult;
            }

            long elapsed = System.currentTimeMillis() - start;
            if (elapsed >= maxWaitMs) {
                lastResult.addProperty("message",
                        "No results found after waiting 10 seconds for indexing. " +
                        "Memories may still be processing.");
                return lastResult;
            }

            try {
                Thread.sleep(pollIntervalMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return lastResult;
            }
        }
    }

    /**
     * List memories within a space, with optional pagination and filtering.
     *
     * @param spaceId          UUID of the space to list memories from
     * @param maxResults       maximum number of memories to return per page, or {@code null} for the server default
     * @param nextToken        pagination token from a previous response, or {@code null} for the first page
     * @param statusFilter     filter by processing status (e.g. "COMPLETED"), or {@code null} for all
     * @param includeContent   whether to include the original content in each memory
     * @param filterExpression server-side filter expression on labels/metadata, or {@code null}
     * @return result envelope containing {@code memories}, {@code totalMemories},
     *         and a {@code nextToken} when more pages are available
     */
    public JsonObject listMemories(String spaceId, Integer maxResults, String nextToken,
                                   String statusFilter, boolean includeContent,
                                   String filterExpression) {
        StringBuilder qs = new StringBuilder();
        if (maxResults != null) {
            appendQuery(qs, "maxResults", String.valueOf(maxResults));
        }
        if (nextToken != null && !nextToken.isEmpty()) {
            appendQuery(qs, "nextToken", urlEncode(nextToken));
        }
        if (statusFilter != null && !statusFilter.isEmpty()) {
            appendQuery(qs, "statusFilter", urlEncode(statusFilter));
        }
        if (includeContent) {
            appendQuery(qs, "includeContent", "true");
        }
        if (filterExpression != null && !filterExpression.isEmpty()) {
            appendQuery(qs, "filter", urlEncode(filterExpression));
        }

        JsonElement body = getJson("/v1/spaces/" + spaceId + "/memories" + qs);

        List<JsonObject> memories;
        String nextTokenResp = null;
        if (body.isJsonObject()) {
            JsonObject obj = body.getAsJsonObject();
            memories = parseListResponse(obj, "memories");
            if (obj.has("nextToken") && !obj.get("nextToken").isJsonNull()) {
                nextTokenResp = obj.get("nextToken").getAsString();
            }
        } else {
            memories = parseListResponse(body, "memories");
        }

        JsonObject result = new JsonObject();
        result.addProperty("success", true);
        result.addProperty("spaceId", spaceId);
        result.add("memories", GSON.toJsonTree(memories));
        result.addProperty("totalMemories", memories.size());
        if (nextTokenResp != null) {
            result.addProperty("nextToken", nextTokenResp);
        }
        return result;
    }

    /**
     * Fetch a specific memory by ID.
     * <p>
     * When {@code includeContent} is true, the original content is returned
     * as JSON when the server responds with {@code application/json}, and as
     * a plain string otherwise (the {@code /content} endpoint returns the
     * raw bytes for non-JSON memories).
     *
     * @param memoryId       UUID of the memory to fetch
     * @param includeContent whether to also fetch the memory's original content
     * @return result envelope containing the {@code memory} object, and {@code content}
     *         or {@code contentError} when {@code includeContent} is true
     */
    public JsonObject getMemory(String memoryId, boolean includeContent) {
        JsonElement memoryBody = getJson("/v1/memories/" + memoryId);

        JsonObject result = new JsonObject();
        result.addProperty("success", true);
        result.add("memory", memoryBody);

        if (includeContent) {
            try {
                HttpResponse<String> response = getRaw("/v1/memories/" + memoryId + "/content");
                String contentType = response.headers()
                        .firstValue("content-type")
                        .orElse("");
                String body = response.body();
                if (contentType.contains("application/json")) {
                    result.add("content", GSON.fromJson(body, JsonElement.class));
                } else {
                    result.addProperty("content", body);
                }
            } catch (Exception e) {
                result.addProperty("contentError", "Failed to fetch content: " + e.getMessage());
            }
        }

        return result;
    }

    /**
     * Delete a memory by ID.
     *
     * @param memoryId UUID of the memory to delete
     * @return result envelope confirming the deletion
     */
    public JsonObject deleteMemory(String memoryId) {
        delete("/v1/memories/" + memoryId);

        JsonObject result = new JsonObject();
        result.addProperty("success", true);
        result.addProperty("memoryId", memoryId);
        result.addProperty("message", "Memory deleted successfully");
        return result;
    }

    /**
     * List all available embedder models.
     *
     * @return all embedder models visible to the authenticated user
     */
    public List<JsonObject> listEmbedders() {
        JsonElement body = getJson("/v1/embedders");
        return parseListResponse(body, "embedders");
    }

    // -- HTTP helpers --

    private HttpClient buildHttpClient() {
        HttpClient.Builder builder = HttpClient.newBuilder()
                .connectTimeout(timeout);
        if (!verifySsl) {
            try {
                javax.net.ssl.SSLContext sslContext = javax.net.ssl.SSLContext.getInstance("TLS");
                sslContext.init(null, new javax.net.ssl.TrustManager[]{
                        new javax.net.ssl.X509TrustManager() {
                            public java.security.cert.X509Certificate[] getAcceptedIssuers() {
                                return new java.security.cert.X509Certificate[0];
                            }
                            public void checkClientTrusted(java.security.cert.X509Certificate[] certs, String authType) {}
                            public void checkServerTrusted(java.security.cert.X509Certificate[] certs, String authType) {}
                        }
                }, new java.security.SecureRandom());
                builder.sslContext(sslContext);
            } catch (Exception e) {
                throw new GoodMemException("Failed to configure SSL context for insecure connections", e);
            }
        }
        return builder.build();
    }

    private JsonElement getJson(String path) {
        return GSON.fromJson(getRaw(path).body(), JsonElement.class);
    }

    private HttpResponse<String> getRaw(String path) {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .header("X-API-Key", apiKey)
                .header("Content-Type", "application/json")
                .header("Accept", "*/*")
                .timeout(timeout)
                .GET()
                .build();

        return execute(request);
    }

    private JsonObject post(String path, JsonObject body) {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .header("X-API-Key", apiKey)
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .timeout(timeout)
                .POST(HttpRequest.BodyPublishers.ofString(GSON.toJson(body)))
                .build();

        HttpResponse<String> response = execute(request);
        return GSON.fromJson(response.body(), JsonObject.class);
    }

    private JsonObject put(String path, JsonObject body) {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .header("X-API-Key", apiKey)
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .timeout(timeout)
                .PUT(HttpRequest.BodyPublishers.ofString(GSON.toJson(body)))
                .build();

        HttpResponse<String> response = execute(request);
        return GSON.fromJson(response.body(), JsonObject.class);
    }

    private String postNdjson(String path, JsonObject body) {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .header("X-API-Key", apiKey)
                .header("Content-Type", "application/json")
                .header("Accept", "application/x-ndjson")
                .timeout(timeout)
                .POST(HttpRequest.BodyPublishers.ofString(GSON.toJson(body)))
                .build();

        HttpResponse<String> response = execute(request);
        return response.body();
    }

    private void delete(String path) {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .header("X-API-Key", apiKey)
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .timeout(timeout)
                .DELETE()
                .build();

        execute(request);
    }

    private HttpResponse<String> execute(HttpRequest request) {
        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() >= 400) {
                throw new GoodMemException(String.format(
                        "GoodMem API error (HTTP %d) for %s %s: %s",
                        response.statusCode(),
                        request.method(),
                        request.uri(),
                        response.body()));
            }

            return response;
        } catch (GoodMemException e) {
            throw e;
        } catch (IOException e) {
            throw new GoodMemException(String.format(
                    "Failed to connect to GoodMem API at %s: %s",
                    request.uri(), e.getMessage()), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new GoodMemException("Request to GoodMem API was interrupted", e);
        }
    }

    private List<JsonObject> parseListResponse(JsonElement body, String arrayKey) {
        List<JsonObject> result = new ArrayList<>();
        JsonArray array;

        if (body.isJsonArray()) {
            array = body.getAsJsonArray();
        } else if (body.isJsonObject() && body.getAsJsonObject().has(arrayKey)) {
            array = body.getAsJsonObject().getAsJsonArray(arrayKey);
        } else {
            return result;
        }

        for (JsonElement element : array) {
            if (element.isJsonObject()) {
                result.add(element.getAsJsonObject());
            }
        }
        return result;
    }

    private static void appendQuery(StringBuilder sb, String key, String value) {
        sb.append(sb.length() == 0 ? '?' : '&').append(key).append('=').append(value);
    }

    private static String urlEncode(String value) {
        return java.net.URLEncoder.encode(value, java.nio.charset.StandardCharsets.UTF_8);
    }
}
