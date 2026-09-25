package ai.pairsys.goodmem.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.pairsys.goodmem.client.Goodmem;
import ai.pairsys.goodmem.client.models.MemoryId;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecifications;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.invocation.InvocationContext;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.service.tool.DefaultToolExecutor;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import okhttp3.mockwebserver.Dispatcher;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.QueueDispatcher;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * The SDK places IDs in URL paths and OkHttp resolves dot segments, so without a guard the memory
 * ID {@code ../spaces/<id>} deletes a space. Every entry point must refuse anything but a UUID
 * before the real SDK sends a request.
 */
class GoodMemIdValidationTest extends SdkTestSupport {
  static final String TARGET = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";

  static final List<Payload> PAYLOADS =
      Stream.of(
              "../spaces/" + TARGET,
              "a/../../spaces/" + TARGET,
              "%2e%2e/spaces/" + TARGET,
              "..%2Fspaces%2F" + TARGET,
              TARGET + "/../../spaces/" + TARGET,
              "",
              " " + TARGET,
              TARGET + "?x=1",
              TARGET + "#frag",
              TARGET + "\n",
              "1-1-1-1-1")
          .map(Payload::new)
          .toList();

  /** Every developer-facing method that accepts an ID. */
  static final List<Entry> ENTRY_POINTS =
      List.of(
          // IDs placed in URL paths.
          new Entry("goodmemGetMemory", "memoryId", (c, id) -> tools(c).goodmemGetMemory(id, true)),
          new Entry(
              "goodmemDeleteMemory",
              "memoryId",
              (c, id) -> {
                tools(c).goodmemDeleteMemory(id);
                return null;
              }),
          new Entry("goodmemGetSpace", "spaceId", (c, id) -> tools(c).goodmemGetSpace(id)),
          new Entry(
              "goodmemUpdateSpace",
              "spaceId",
              (c, id) -> tools(c).goodmemUpdateSpace(id, "renamed", null, null)),
          new Entry(
              "goodmemDeleteSpace",
              "spaceId",
              (c, id) -> {
                tools(c).goodmemDeleteSpace(id);
                return null;
              }),
          new Entry(
              "goodmemListMemories",
              "spaceId",
              (c, id) -> tools(c).goodmemListMemories(id, null, null, null)),
          new Entry(
              "GoodMemIndexing.waitForMemory",
              "memoryId",
              (c, id) ->
                  GoodMemIndexing.waitForMemory(c, MemoryId.from(id), Duration.ofSeconds(1))),
          // IDs sent in request bodies or configured once.
          new Entry(
              "goodmemCreateSpace",
              "embedderId",
              (c, id) -> tools(c).goodmemCreateSpace("n", id, null)),
          new Entry(
              "goodmemCreateMemory",
              "spaceId",
              (c, id) -> tools(c).goodmemCreateMemory(id, "text", null, null, false)),
          new Entry(
              "goodmemRetrieveMemories spaceIds",
              "spaceIds",
              (c, id) ->
                  tools(c).goodmemRetrieveMemories("q", List.of(id), null, null, null, null, null)),
          new Entry(
              "goodmemRetrieveMemories rerankerId",
              "rerankerId",
              (c, id) ->
                  tools(c)
                      .goodmemRetrieveMemories("q", List.of(SPACE), null, null, null, id, null)),
          new Entry(
              "goodmemRetrieveMemories llmId",
              "llmId",
              (c, id) ->
                  tools(c)
                      .goodmemRetrieveMemories("q", List.of(SPACE), null, null, null, null, id)),
          new Entry(
              "GoodMemContentRetriever spaceIds",
              "spaceIds",
              (c, id) ->
                  GoodMemContentRetriever.builder()
                      .client(c)
                      .spaceIds(List.of(id))
                      .build()
                      .retrieve(Query.from("q"))),
          new Entry(
              "GoodMemContentRetriever rerankerId",
              "rerankerId",
              (c, id) ->
                  GoodMemContentRetriever.builder()
                      .client(c)
                      .spaceIds(List.of(SPACE))
                      .rerankerId(id)
                      .build()
                      .retrieve(Query.from("q"))),
          new Entry(
              "GoodMemDocumentIngestor spaceId",
              "spaceId",
              (c, id) ->
                  GoodMemDocumentIngestor.builder()
                      .client(c)
                      .spaceId(id)
                      .build()
                      .ingest(List.of(Document.from("text")))),
          new Entry(
              "GoodMemIndexing.waitForMemories",
              "memoryIds",
              (c, id) -> {
                GoodMemIndexing.waitForMemories(
                    c, List.of(MemoryId.from(id)), Duration.ofSeconds(1));
                return null;
              }));

  /** Every model-visible ID argument, invoked through LangChain4j's own tool executor. */
  static final List<ToolArgument> TOOL_ARGUMENTS =
      List.of(
          new ToolArgument("goodmemGetMemory", "memoryId", id -> Map.of("memoryId", id)),
          new ToolArgument("goodmemDeleteMemory", "memoryId", id -> Map.of("memoryId", id)),
          new ToolArgument("goodmemGetSpace", "spaceId", id -> Map.of("spaceId", id)),
          new ToolArgument(
              "goodmemUpdateSpace", "spaceId", id -> Map.of("spaceId", id, "name", "renamed")),
          new ToolArgument("goodmemDeleteSpace", "spaceId", id -> Map.of("spaceId", id)),
          new ToolArgument("goodmemListMemories", "spaceId", id -> Map.of("spaceId", id)),
          new ToolArgument(
              "goodmemCreateSpace", "embedderId", id -> Map.of("name", "n", "embedderId", id)),
          new ToolArgument(
              "goodmemCreateMemory",
              "spaceId",
              id -> Map.of("spaceId", id, "textContent", "text", "wait", false)),
          new ToolArgument(
              "goodmemRetrieveMemories",
              "spaceIds",
              id -> Map.of("query", "q", "spaceIds", List.of(id))),
          new ToolArgument(
              "goodmemRetrieveMemories",
              "rerankerId",
              id -> Map.of("query", "q", "spaceIds", List.of(SPACE), "rerankerId", id)),
          new ToolArgument(
              "goodmemRetrieveMemories",
              "llmId",
              id -> Map.of("query", "q", "spaceIds", List.of(SPACE), "llmId", id)));

  @BeforeEach
  void answerEveryRequest() {
    // Lets unguarded code complete quickly, so a failure reports what the server received.
    server.setDispatcher(
        new Dispatcher() {
          @Override
          public MockResponse dispatch(RecordedRequest request) {
            return new MockResponse()
                .setHeader("Content-Type", "application/json")
                .setBody(
                    json(
                        Map.of(
                            "memoryId",
                            TARGET,
                            "spaceId",
                            TARGET,
                            "name",
                            "fixture",
                            "processingStatus",
                            "COMPLETED",
                            "memories",
                            List.of())));
          }
        });
  }

  static Stream<Arguments> entryPointsAndPayloads() {
    return ENTRY_POINTS.stream()
        .flatMap(entry -> PAYLOADS.stream().map(payload -> Arguments.of(entry, payload)));
  }

  @ParameterizedTest(name = "{0} refuses {1}")
  @MethodSource("entryPointsAndPayloads")
  void developerEntryPointsRefuseNonUuidIdsWithoutARequest(Entry entry, Payload payload)
      throws Exception {
    Throwable thrown = null;
    try {
      entry.call().run(client, payload.value());
    } catch (Exception e) {
      thrown = e;
    }
    assertEquals(List.of(), received(), entry + " sent a request for " + payload);
    assertInstanceOf(IllegalArgumentException.class, thrown, entry + " accepted " + payload);
    assertTrue(
        thrown.getMessage().startsWith(entry.field() + " must be a UUID"), thrown.getMessage());
  }

  static Stream<Arguments> toolArgumentsAndPayloads() {
    return TOOL_ARGUMENTS.stream()
        .flatMap(argument -> PAYLOADS.stream().map(payload -> Arguments.of(argument, payload)));
  }

  @ParameterizedTest(name = "{0} refuses {1}")
  @MethodSource("toolArgumentsAndPayloads")
  void modelToolCallsRefuseNonUuidIdsWithoutARequest(ToolArgument argument, Payload payload)
      throws Exception {
    var invocation =
        ToolExecutionRequest.builder()
            .id("call")
            .name(argument.tool())
            .arguments(json(argument.arguments().apply(payload.value())))
            .build();
    var result =
        new DefaultToolExecutor(new GoodMemTools(client), invocation)
            .executeWithContext(invocation, InvocationContext.builder().build());
    assertEquals(List.of(), received(), argument + " sent a request for " + payload);
    assertTrue(result.isError(), argument + " reported success: " + result.resultText());
    assertTrue(
        result.resultText().startsWith(argument.field() + " must be a UUID"), result.resultText());
  }

  static Stream<Arguments> routes() {
    return Stream.of(TARGET, TARGET.toUpperCase(Locale.ROOT))
        .flatMap(
            id ->
                Stream.of(
                    Arguments.of("goodmemGetMemory", id, "GET /v1/memories/" + TARGET),
                    Arguments.of("goodmemDeleteMemory", id, "DELETE /v1/memories/" + TARGET),
                    Arguments.of("goodmemGetSpace", id, "GET /v1/spaces/" + TARGET),
                    Arguments.of("goodmemUpdateSpace", id, "PUT /v1/spaces/" + TARGET),
                    Arguments.of("goodmemDeleteSpace", id, "DELETE /v1/spaces/" + TARGET),
                    Arguments.of(
                        "goodmemListMemories", id, "GET /v1/spaces/" + TARGET + "/memories"),
                    Arguments.of(
                        "GoodMemIndexing.waitForMemory", id, "GET /v1/memories/" + TARGET)));
  }

  @ParameterizedTest(name = "{0}({1}) sends {2}")
  @MethodSource("routes")
  void aUuidReachesExactlyTheIntendedPathInLowercase(String name, String id, String expected)
      throws Exception {
    ENTRY_POINTS.stream()
        .filter(entry -> entry.name().equals(name))
        .findFirst()
        .orElseThrow()
        .call()
        .run(client, id);
    assertEquals(List.of(expected), received());
  }

  @Test
  void configuredUppercaseSpaceIdsAreNormalisedAndMatchServerIds() throws Exception {
    server.setDispatcher(new QueueDispatcher());
    events(chunk(CHUNK, MEMORY, "Text", 0.5), definition(MEMORY, TARGET, Map.of()));
    var contents =
        GoodMemContentRetriever.builder()
            .client(client)
            .spaceIds(List.of(TARGET.toUpperCase(Locale.ROOT)))
            .build()
            .retrieve(Query.from("q"));
    assertEquals("Text", contents.getFirst().textSegment().text());
    assertEquals(TARGET, body(request()).at("/spaceKeys/0/spaceId").asText());
  }

  @Test
  void aNonUuidMemoryIdFromTheServerIsNotPolled() throws Exception {
    server.setDispatcher(new QueueDispatcher());
    reply(json(memory("../spaces/" + TARGET, SPACE, "PENDING", Map.of())));
    Throwable thrown = null;
    try {
      tools(client).goodmemCreateMemory(SPACE, "text", null, null, null);
    } catch (RuntimeException e) {
      thrown = e;
    }
    assertEquals(List.of("POST /v1/memories"), received());
    assertInstanceOf(IllegalArgumentException.class, thrown);
  }

  @Test
  void everyModelVisibleIdIsDescribedAsAUuidAndCoveredAbove() {
    Set<String> visible = new TreeSet<>();
    for (var spec : ToolSpecifications.toolSpecificationsFrom(GoodMemTools.class)) {
      if (spec.parameters() == null) {
        continue;
      }
      spec.parameters()
          .properties()
          .forEach(
              (name, schema) -> {
                if (name.endsWith("Id") || name.endsWith("Ids")) {
                  visible.add(spec.name() + "." + name);
                  assertTrue(
                      schema.description().contains("UUID"),
                      spec.name() + "." + name + ": " + schema.description());
                }
              });
    }
    assertEquals(
        visible,
        TOOL_ARGUMENTS.stream()
            .map(ToolArgument::toString)
            .collect(Collectors.toCollection(TreeSet::new)));
    assertFalse(visible.isEmpty());
  }

  private List<String> received() throws Exception {
    List<String> requests = new ArrayList<>();
    for (int i = server.getRequestCount(); i > 0; i--) {
      RecordedRequest request = server.takeRequest(1, TimeUnit.SECONDS);
      if (request != null) {
        requests.add(request.getMethod() + " " + request.getRequestUrl().encodedPath());
      }
    }
    return requests;
  }

  private static GoodMemTools tools(Goodmem client) {
    return new GoodMemTools(client);
  }

  /** Invokes one entry point with the given ID. */
  interface Call {
    Object run(Goodmem client, String id) throws Exception;
  }

  record Entry(String name, String field, Call call) {
    @Override
    public String toString() {
      return name;
    }
  }

  record ToolArgument(String tool, String field, Function<String, Map<String, Object>> arguments) {
    @Override
    public String toString() {
      return tool + "." + field;
    }
  }

  record Payload(String value) {
    @Override
    public String toString() {
      return json(value);
    }
  }
}
