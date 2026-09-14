package ai.pairsys.goodmem.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.service.AiServices;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class GoodMemSearchToolTest extends SdkTestSupport {
  interface Assistant {
    String chat(String question);
  }

  @Test
  void twoNamedScopedSearchesWorkInOneAiService() throws Exception {
    var policies = retriever().build().asTool("search_policies", "Search company return policies");
    var tickets =
        retriever()
            .spaceIds(List.of(SPACE_2))
            .build()
            .asTool("search_tickets", "Search customer support tickets");
    events(
        chunk(CHUNK, MEMORY, "Policy text", 0.9),
        definition(MEMORY, SPACE, Map.of("source", "policies")));
    events(
        chunk(CHUNK_2, MEMORY_2, "Ticket text", 0.8),
        definition(MEMORY_2, SPACE_2, Map.of("source", "tickets")));
    var calls = new AtomicInteger();
    ChatModel model =
        new ChatModel() {
          @Override
          public ChatResponse doChat(ChatRequest request) {
            if (calls.getAndIncrement() == 0) {
              assertEquals(
                  List.of("search_policies", "search_tickets"),
                  request.toolSpecifications().stream().map(s -> s.name()).sorted().toList());
              assertTrue(
                  request.toolSpecifications().stream()
                      .anyMatch(s -> s.description().contains("return policies")));
              return ChatResponse.builder()
                  .aiMessage(
                      AiMessage.from(
                          List.of(
                              ToolExecutionRequest.builder()
                                  .id("p")
                                  .name("search_policies")
                                  .arguments("{\"query\":\"return policy\"}")
                                  .build(),
                              ToolExecutionRequest.builder()
                                  .id("t")
                                  .name("search_tickets")
                                  .arguments("{\"query\":\"return ticket\"}")
                                  .build())))
                  .build();
            }
            assertTrue(request.messages().toString().contains("Policy text"));
            assertTrue(request.messages().toString().contains("Ticket text"));
            return ChatResponse.builder().aiMessage(AiMessage.from("Found both sources")).build();
          }
        };
    var searches = List.of(policies, tickets);
    var assistant = AiServices.builder(Assistant.class).chatModel(model).tools(searches).build();
    assertEquals("Found both sources", assistant.chat("Find the policy and a ticket"));
    assertEquals(SPACE, body(request()).at("/spaceKeys/0/spaceId").asText());
    assertEquals(SPACE_2, body(request()).at("/spaceKeys/0/spaceId").asText());
  }

  @Test
  void descriptionsAndNamesAreRequired() {
    var retriever = retriever().build();
    assertThrows(IllegalArgumentException.class, () -> retriever.asTool("bad name", "description"));
    assertThrows(IllegalArgumentException.class, () -> retriever.asTool("search", " "));
  }
}
