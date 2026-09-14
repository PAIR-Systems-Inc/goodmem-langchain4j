package ai.pairsys.goodmem.langchain4j;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.service.tool.AiServiceTool;
import java.util.Map;

/** Builds native AI Service tools bound to application-configured retrieval. */
final class GoodMemSearchTool {
  private static final ObjectMapper JSON = new ObjectMapper();

  private GoodMemSearchTool() {}

  static AiServiceTool create(GoodMemContentRetriever retriever, String name, String description) {
    if (name == null || !name.matches("[a-zA-Z0-9_-]{1,64}")) {
      throw new IllegalArgumentException(
          "Tool name must contain 1–64 letters, digits, underscores or hyphens");
    }
    if (description == null || description.isBlank()) {
      throw new IllegalArgumentException("Tool description must not be blank");
    }
    var specification =
        ToolSpecification.builder()
            .name(name)
            .description(description)
            .parameters(
                JsonObjectSchema.builder()
                    .addStringProperty("query", "A focused natural-language search query")
                    .required("query")
                    .additionalProperties(false)
                    .build())
            .build();
    return AiServiceTool.builder()
        .toolSpecification(specification)
        .toolExecutor(
            (request, memoryId) -> {
              try {
                var arguments = JSON.readTree(request.arguments());
                if (arguments == null
                    || !arguments.isObject()
                    || arguments.size() != 1
                    || !arguments.path("query").isTextual()
                    || arguments.path("query").asText().isBlank()) {
                  throw new IllegalArgumentException("Search expects only a nonblank query string");
                }
                var results = retriever.retrieve(Query.from(arguments.get("query").asText()));
                return JSON.writeValueAsString(
                    results.stream()
                        .map(
                            content ->
                                Map.of(
                                    "text",
                                    content.textSegment().text(),
                                    "metadata",
                                    content.textSegment().metadata().toMap()))
                        .toList());
              } catch (JsonProcessingException e) {
                throw new IllegalArgumentException("Cannot encode or decode search tool JSON", e);
              }
            })
        .build();
  }
}
