package ai.pairsys.goodmem.langchain4j;

import static org.junit.jupiter.api.Assertions.assertNotNull;

import ai.pairsys.goodmem.client.Goodmem;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;

/** Every HTTP request is made by the real, published SDK. */
abstract class SdkTestSupport {
  static final String SPACE = "11111111-1111-4111-8111-111111111111";
  static final String SPACE_2 = "11111111-1111-4111-8111-111111111112";
  static final String MEMORY = "22222222-2222-4222-8222-222222222222";
  static final String MEMORY_2 = "22222222-2222-4222-8222-222222222223";
  static final String CHUNK = "33333333-3333-4333-8333-333333333333";
  static final String CHUNK_2 = "33333333-3333-4333-8333-333333333334";
  static final String EMBEDDER = "44444444-4444-4444-8444-444444444444";
  static final String RERANKER = "55555555-5555-4555-8555-555555555555";
  static final ObjectMapper JSON = new ObjectMapper();
  MockWebServer server;
  Goodmem client;

  @BeforeEach
  void startSdk() throws Exception {
    server = new MockWebServer();
    server.start();
    client =
        Goodmem.builder()
            .baseUrl(server.url("/").toString())
            .apiKey("fixture-key")
            .timeout(Duration.ofSeconds(2))
            .build();
  }

  @AfterEach
  void stopSdk() throws Exception {
    client.close();
    server.shutdown();
  }

  GoodMemContentRetriever.Builder retriever() {
    return GoodMemContentRetriever.builder().client(client).spaceIds(List.of(SPACE));
  }

  void reply(String body) {
    server.enqueue(new MockResponse().setHeader("Content-Type", "application/json").setBody(body));
  }

  void events(String... lines) {
    server.enqueue(
        new MockResponse()
            .setHeader("Content-Type", "application/x-ndjson")
            .setBody(String.join("\n", lines) + "\n"));
  }

  RecordedRequest request() throws Exception {
    var request = server.takeRequest(1, TimeUnit.SECONDS);
    assertNotNull(request, "Expected an SDK request");
    return request;
  }

  static JsonNode body(RecordedRequest request) throws Exception {
    return JSON.readTree(request.getBody().readUtf8());
  }

  static String json(Object object) {
    try {
      return JSON.writeValueAsString(object);
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  static Map<String, Object> memory(
      String id, String space, String status, Map<String, ?> metadata) {
    return Map.of(
        "memoryId",
        id,
        "spaceId",
        space,
        "processingStatus",
        status,
        "contentType",
        "text/plain",
        "metadata",
        metadata);
  }

  static String definition(String id, String space, Map<String, ?> metadata) {
    return json(Map.of("memoryDefinition", memory(id, space, "COMPLETED", metadata)));
  }

  static String chunk(String id, String memory, String text, double score) {
    return json(
        Map.of(
            "retrievedItem",
            Map.of(
                "chunk",
                Map.of(
                    "chunk",
                    Map.of("chunkId", id, "memoryId", memory, "chunkText", text),
                    "relevanceScore",
                    score))));
  }

  static String status(String code) {
    return json(Map.of("status", Map.of("code", code, "message", "Fixture diagnostic")));
  }

  static String space(String id) {
    return json(Map.of("spaceId", id, "name", "Example space"));
  }

  static String batchMemories(List<Map<String, Object>> memories) {
    return json(
        Map.of(
            "results", memories.stream().map(m -> Map.of("success", true, "memory", m)).toList()));
  }
}
