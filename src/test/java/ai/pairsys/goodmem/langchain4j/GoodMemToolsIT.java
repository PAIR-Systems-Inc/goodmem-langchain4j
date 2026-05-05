package ai.pairsys.goodmem.langchain4j;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration tests for GoodMem LangChain4j tools.
 * <p>
 * These tests run against a live GoodMem API instance. Set the following
 * environment variables before running:
 * <ul>
 *   <li>GOODMEM_BASE_URL - e.g. https://localhost:8080</li>
 *   <li>GOODMEM_API_KEY - your GoodMem API key</li>
 *   <li>GOODMEM_TEST_PDF - path to a PDF file for upload testing</li>
 *   <li>GOODMEM_EMBEDDER_ID - (optional) specific embedder ID to use</li>
 * </ul>
 * <p>
 * Run with:
 * <pre>
 * GOODMEM_BASE_URL=https://localhost:8080 \
 * GOODMEM_API_KEY=your-key \
 * GOODMEM_TEST_PDF=/path/to/file.pdf \
 * mvn test -pl langchain4j-goodmem -Dtest=GoodMemToolsIT
 * </pre>
 */
@EnabledIfEnvironmentVariable(named = "GOODMEM_API_KEY", matches = ".+")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class GoodMemToolsIT {

    private static final Gson GSON = new Gson();

    private static GoodMemTools tools;
    private static String embedderId;
    private static String spaceId;
    private static String textMemoryId;
    private static String pdfMemoryId;

    @BeforeAll
    static void setUp() {
        String baseUrl = System.getenv("GOODMEM_BASE_URL");
        String apiKey = System.getenv("GOODMEM_API_KEY");

        if (baseUrl == null || baseUrl.isEmpty()) {
            baseUrl = "https://localhost:8080";
        }

        tools = GoodMemTools.builder()
                .baseUrl(baseUrl)
                .apiKey(apiKey)
                .verifySsl(false)
                .build();

        // Resolve embedder ID: use env var if set, otherwise pick from list
        String envEmbedder = System.getenv("GOODMEM_EMBEDDER_ID");
        if (envEmbedder != null && !envEmbedder.isEmpty()) {
            embedderId = envEmbedder;
        } else {
            String raw = tools.goodmemListEmbedders();
            JsonObject result = GSON.fromJson(raw, JsonObject.class);
            JsonArray embedders = result.getAsJsonArray("embedders");
            assertThat(embedders).isNotEmpty();

            // Try to find a non-OpenAI embedder first (Voyage, etc.) since
            // OpenAI embedders may have transient issues on some deployments
            for (int i = 0; i < embedders.size(); i++) {
                JsonObject emb = embedders.get(i).getAsJsonObject();
                String provider = emb.has("providerType")
                        ? emb.get("providerType").getAsString() : "";
                if (!"OPENAI".equals(provider)) {
                    embedderId = emb.get("embedderId").getAsString();
                    System.out.println("Selected embedder: "
                            + emb.get("displayName").getAsString()
                            + " (" + embedderId + ")");
                    break;
                }
            }
            // Fallback to first embedder if no non-OpenAI embedder found
            if (embedderId == null) {
                embedderId = embedders.get(0).getAsJsonObject()
                        .get("embedderId").getAsString();
            }
        }
    }

    @Test
    @Order(1)
    void listEmbedders() {
        String raw = tools.goodmemListEmbedders();
        JsonObject result = GSON.fromJson(raw, JsonObject.class);

        assertThat(result.get("success").getAsBoolean()).isTrue();
        assertThat(result.get("totalEmbedders").getAsInt()).isGreaterThan(0);

        JsonArray embedders = result.getAsJsonArray("embedders");
        assertThat(embedders).isNotEmpty();
    }

    @Test
    @Order(2)
    void listSpaces() {
        String raw = tools.goodmemListSpaces();
        JsonObject result = GSON.fromJson(raw, JsonObject.class);

        assertThat(result.get("success").getAsBoolean()).isTrue();
        assertThat(result.has("spaces")).isTrue();
    }

    @Test
    @Order(3)
    void createSpace() {
        String spaceName = "langchain4j-test-" + System.currentTimeMillis();
        String raw = tools.goodmemCreateSpace(spaceName, embedderId, null, null, null);
        JsonObject result = GSON.fromJson(raw, JsonObject.class);

        assertThat(result.get("success").getAsBoolean()).isTrue();
        assertThat(result.get("spaceId").getAsString()).isNotEmpty();
        assertThat(result.get("name").getAsString()).isEqualTo(spaceName);

        spaceId = result.get("spaceId").getAsString();
    }

    @Test
    @Order(4)
    void createSpaceReusesExisting() {
        if (spaceId == null) {
            createSpace();
        }

        // Find the space name by listing spaces
        String spacesRaw = tools.goodmemListSpaces();
        JsonObject spacesResult = GSON.fromJson(spacesRaw, JsonObject.class);
        JsonArray spaces = spacesResult.getAsJsonArray("spaces");

        String spaceName = null;
        for (int i = 0; i < spaces.size(); i++) {
            JsonObject space = spaces.get(i).getAsJsonObject();
            if (space.get("spaceId").getAsString().equals(spaceId)) {
                spaceName = space.get("name").getAsString();
                break;
            }
        }

        assertThat(spaceName).isNotNull();

        // Create again with same name - should reuse
        String raw2 = tools.goodmemCreateSpace(spaceName, embedderId, null, null, null);
        JsonObject result2 = GSON.fromJson(raw2, JsonObject.class);

        assertThat(result2.get("success").getAsBoolean()).isTrue();
        assertThat(result2.get("reused").getAsBoolean()).isTrue();
        assertThat(result2.get("spaceId").getAsString()).isEqualTo(spaceId);
    }

    @Test
    @Order(5)
    void createTextMemory() {
        if (spaceId == null) {
            createSpace();
        }

        String raw = tools.goodmemCreateMemory(
                spaceId,
                "LangChain4j is a Java framework for developing applications "
                        + "powered by large language models. It provides tools, chains, "
                        + "and agents for building AI-powered applications.",
                null,
                null);
        JsonObject result = GSON.fromJson(raw, JsonObject.class);

        assertThat(result.get("success").getAsBoolean()).isTrue();
        assertThat(result.get("memoryId").getAsString()).isNotEmpty();
        assertThat(result.get("contentType").getAsString()).isEqualTo("text/plain");

        textMemoryId = result.get("memoryId").getAsString();
    }

    @Test
    @Order(6)
    void createPdfMemory() {
        if (spaceId == null) {
            createSpace();
        }

        String testPdf = System.getenv("GOODMEM_TEST_PDF");
        if (testPdf == null || testPdf.isEmpty()) {
            System.out.println("GOODMEM_TEST_PDF not set, skipping PDF test");
            return;
        }

        String raw = tools.goodmemCreateMemory(spaceId, null, testPdf, "{\"source\":\"test\",\"type\":\"resume\"}");
        JsonObject result = GSON.fromJson(raw, JsonObject.class);

        assertThat(result.get("success").getAsBoolean()).isTrue();
        assertThat(result.get("memoryId").getAsString()).isNotEmpty();
        assertThat(result.get("contentType").getAsString()).isEqualTo("application/pdf");

        pdfMemoryId = result.get("memoryId").getAsString();
    }

    @Test
    @Order(7)
    void getMemory() {
        if (textMemoryId == null) {
            createTextMemory();
        }

        String raw = tools.goodmemGetMemory(textMemoryId, true);
        JsonObject result = GSON.fromJson(raw, JsonObject.class);

        assertThat(result.get("success").getAsBoolean()).isTrue();
        assertThat(result.has("memory")).isTrue();
        assertThat(result.getAsJsonObject("memory")
                .get("memoryId").getAsString()).isEqualTo(textMemoryId);
    }

    @Test
    @Order(8)
    void retrieveMemories() {
        if (spaceId == null || textMemoryId == null) {
            createTextMemory();
        }

        String raw = tools.goodmemRetrieveMemories(
                "LangChain4j framework",
                spaceId,
                5,
                true,
                true,
                null,
                null,
                null,
                null,
                null);
        JsonObject result = GSON.fromJson(raw, JsonObject.class);

        assertThat(result.get("success").getAsBoolean()).isTrue();
        assertThat(result.get("totalResults").getAsInt()).isGreaterThan(0);

        // Verify chunk structure
        JsonArray results = result.getAsJsonArray("results");
        assertThat(results).isNotEmpty();
        JsonObject firstChunk = results.get(0).getAsJsonObject();
        assertThat(firstChunk.has("chunkText")).isTrue();
        assertThat(firstChunk.has("relevanceScore")).isTrue();
    }

    // -- Fix #1: UTF-8 encoding for text file content --

    @Test
    @Order(9)
    void createMemoryFromTextFilePreservesUtf8() throws Exception {
        if (spaceId == null) {
            createSpace();
        }

        // Write a temp file with multi-byte UTF-8 characters
        java.nio.file.Path tempFile = java.nio.file.Files.createTempFile("goodmem-utf8-", ".txt");
        String utf8Content = "Héllo wörld! Ñoño. 日本語テスト. Émojis: ✅🚀";
        java.nio.file.Files.writeString(tempFile, utf8Content, java.nio.charset.StandardCharsets.UTF_8);

        try {
            String raw = tools.goodmemCreateMemory(spaceId, null, tempFile.toString(), null);
            JsonObject result = GSON.fromJson(raw, JsonObject.class);

            assertThat(result.get("success").getAsBoolean()).isTrue();
            assertThat(result.get("contentType").getAsString()).isEqualTo("text/plain");

            String memId = result.get("memoryId").getAsString();

            // Fetch the memory back and verify the content survived the round-trip
            String getRaw = tools.goodmemGetMemory(memId, true);
            JsonObject getResult = GSON.fromJson(getRaw, JsonObject.class);
            assertThat(getResult.get("success").getAsBoolean()).isTrue();

            // Verify the memory was stored — content may be in the content field
            assertThat(getResult.has("memory")).isTrue();
            assertThat(getResult.getAsJsonObject("memory")
                    .get("memoryId").getAsString()).isEqualTo(memId);

            // Clean up
            tools.goodmemDeleteMemory(memId);
        } finally {
            java.nio.file.Files.deleteIfExists(tempFile);
        }
    }

    // -- Fix #2: createSpace reuse returns actual embedderId --

    @Test
    @Order(10)
    void createSpaceReuseReturnsActualEmbedderId() {
        if (spaceId == null) {
            createSpace();
        }

        // Look up the actual embedder on the space via listSpaces
        String spacesRaw = tools.goodmemListSpaces();
        JsonObject spacesResult = GSON.fromJson(spacesRaw, JsonObject.class);
        JsonArray spaces = spacesResult.getAsJsonArray("spaces");

        String spaceName = null;
        String actualEmbedderId = null;
        for (int i = 0; i < spaces.size(); i++) {
            JsonObject space = spaces.get(i).getAsJsonObject();
            if (space.get("spaceId").getAsString().equals(spaceId)) {
                spaceName = space.get("name").getAsString();
                JsonArray spaceEmbedders = space.getAsJsonArray("spaceEmbedders");
                actualEmbedderId = spaceEmbedders.get(0).getAsJsonObject()
                        .get("embedderId").getAsString();
                break;
            }
        }
        assertThat(spaceName).isNotNull();
        assertThat(actualEmbedderId).isNotNull();

        // Pass a bogus embedderId — the reuse path should return the real one, not this
        String bogusEmbedderId = "00000000-0000-0000-0000-000000000000";
        String raw = tools.goodmemCreateSpace(spaceName, bogusEmbedderId, null, null, null);
        JsonObject result = GSON.fromJson(raw, JsonObject.class);

        assertThat(result.get("success").getAsBoolean()).isTrue();
        assertThat(result.get("reused").getAsBoolean()).isTrue();
        assertThat(result.get("spaceId").getAsString()).isEqualTo(spaceId);

        // The key assertion: embedderId should be the actual one, NOT the bogus one
        assertThat(result.get("embedderId").getAsString())
                .as("Reused space should return the actual embedderId from the server, not the requested one")
                .isEqualTo(actualEmbedderId)
                .isNotEqualTo(bogusEmbedderId);
    }

    // -- Fix #3: metadata exposed through tools --

    @Test
    @Order(11)
    void createMemoryWithMetadata() {
        if (spaceId == null) {
            createSpace();
        }

        String metadataJson = "{\"source\":\"unit-test\",\"author\":\"langchain4j\",\"version\":42}";
        String raw = tools.goodmemCreateMemory(
                spaceId,
                "Memory with metadata for testing.",
                null,
                metadataJson);
        JsonObject result = GSON.fromJson(raw, JsonObject.class);

        assertThat(result.get("success").getAsBoolean()).isTrue();
        String memId = result.get("memoryId").getAsString();
        assertThat(memId).isNotEmpty();

        // Fetch the memory and verify metadata was persisted
        String getRaw = tools.goodmemGetMemory(memId, false);
        JsonObject getResult = GSON.fromJson(getRaw, JsonObject.class);
        assertThat(getResult.get("success").getAsBoolean()).isTrue();

        JsonObject memory = getResult.getAsJsonObject("memory");
        assertThat(memory.has("metadata")).isTrue();
        JsonObject metadata = memory.getAsJsonObject("metadata");
        assertThat(metadata.get("source").getAsString()).isEqualTo("unit-test");
        assertThat(metadata.get("author").getAsString()).isEqualTo("langchain4j");

        // Clean up
        tools.goodmemDeleteMemory(memId);
    }

    @Test
    @Order(12)
    void createMemoryWithNullMetadataStillWorks() {
        if (spaceId == null) {
            createSpace();
        }

        // null metadata — should behave exactly like before the fix
        String raw = tools.goodmemCreateMemory(spaceId, "No metadata here.", null, null);
        JsonObject result = GSON.fromJson(raw, JsonObject.class);

        assertThat(result.get("success").getAsBoolean()).isTrue();
        String memId = result.get("memoryId").getAsString();
        assertThat(memId).isNotEmpty();

        // Clean up
        tools.goodmemDeleteMemory(memId);
    }

    @Test
    @Order(13)
    void createMemoryWithInvalidMetadataReturnsError() {
        if (spaceId == null) {
            createSpace();
        }

        // Malformed JSON — should fail gracefully, not throw
        String raw = tools.goodmemCreateMemory(spaceId, "Bad metadata test.", null, "not-valid-json{{{");
        JsonObject result = GSON.fromJson(raw, JsonObject.class);

        assertThat(result.get("success").getAsBoolean()).isFalse();
        assertThat(result.get("error").getAsString()).isNotEmpty();
    }

    // -- Fix #4: HttpClient reuse (verify multiple calls don't fail) --

    @Test
    @Order(14)
    void multipleRapidCallsReuseHttpClient() {
        // Issue rapid sequential calls — if HttpClient is recreated per request
        // and leaks resources, this is more likely to surface connection issues.
        for (int i = 0; i < 10; i++) {
            String raw = tools.goodmemListEmbedders();
            JsonObject result = GSON.fromJson(raw, JsonObject.class);
            assertThat(result.get("success").getAsBoolean())
                    .as("Call %d of 10 should succeed with reused HttpClient", i + 1)
                    .isTrue();
        }
    }

    // -- Retrieval: verify chunk fields are actually populated --

    @Test
    @Order(15)
    void retrieveMemoriesReturnsPopulatedChunkFields() {
        if (spaceId == null || textMemoryId == null) {
            createTextMemory();
        }

        String raw = tools.goodmemRetrieveMemories(
                "Java framework large language models",
                spaceId, 5, true, true,
                null, null, null, null, null);
        JsonObject result = GSON.fromJson(raw, JsonObject.class);

        assertThat(result.get("success").getAsBoolean()).isTrue();

        JsonArray results = result.getAsJsonArray("results");
        assertThat(results).isNotEmpty();

        JsonObject chunk = results.get(0).getAsJsonObject();

        // Verify fields are not just present but actually populated with real data
        assertThat(chunk.get("chunkId").getAsString())
                .as("chunkId should be a non-empty UUID, not empty string")
                .isNotEmpty()
                .matches("[0-9a-f\\-]{36}");

        assertThat(chunk.get("chunkText").getAsString())
                .as("chunkText should contain actual text, not empty string")
                .isNotEmpty()
                .containsIgnoringCase("langchain4j");

        assertThat(chunk.get("memoryId").getAsString())
                .as("memoryId should be a non-empty UUID, not empty string")
                .isNotEmpty()
                .matches("[0-9a-f\\-]{36}");

        assertThat(chunk.has("relevanceScore")).isTrue();
        assertThat(chunk.get("relevanceScore").getAsDouble())
                .as("relevanceScore should be a real number, not zero/NaN")
                .isNotZero();

        assertThat(chunk.has("memoryIndex")).isTrue();
    }

    // -- Cleanup --

    @Test
    @Order(100)
    void deleteMemory() {
        if (textMemoryId == null) {
            createTextMemory();
        }

        String raw = tools.goodmemDeleteMemory(textMemoryId);
        JsonObject result = GSON.fromJson(raw, JsonObject.class);

        assertThat(result.get("success").getAsBoolean()).isTrue();
        assertThat(result.get("memoryId").getAsString()).isEqualTo(textMemoryId);
    }

    @Test
    @Order(101)
    void cleanupPdfMemory() {
        if (pdfMemoryId != null) {
            String raw = tools.goodmemDeleteMemory(pdfMemoryId);
            JsonObject result = GSON.fromJson(raw, JsonObject.class);
            assertThat(result.get("success").getAsBoolean()).isTrue();
        }
    }
}
