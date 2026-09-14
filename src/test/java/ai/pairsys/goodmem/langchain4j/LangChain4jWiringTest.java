package ai.pairsys.goodmem.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.rag.AugmentationRequest;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.listener.ContentRetrieverErrorContext;
import dev.langchain4j.rag.content.retriever.listener.ContentRetrieverListener;
import dev.langchain4j.rag.content.retriever.listener.ContentRetrieverRequestContext;
import dev.langchain4j.rag.content.retriever.listener.ContentRetrieverResponseContext;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Result;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class LangChain4jWiringTest extends SdkTestSupport {
  interface Assistant {
    Result<String> chat(String question);
  }

  @Test
  void aiServicesInjectsRetrievedTextAndExposesSourceDocuments() {
    events(
        chunk(CHUNK, MEMORY, "Orchid is the launch code.", 0.8),
        definition(MEMORY, SPACE, Map.of("source", "https://example.com/launch")));
    ChatModel model =
        new ChatModel() {
          @Override
          public ChatResponse doChat(ChatRequest request) {
            assertTrue(request.messages().toString().contains("Orchid is the launch code."));
            return ChatResponse.builder()
                .aiMessage(AiMessage.from("The launch code is Orchid."))
                .build();
          }
        };
    var assistant =
        AiServices.builder(Assistant.class)
            .chatModel(model)
            .contentRetriever(retriever().build())
            .build();
    var answer = assistant.chat("What is the launch code?");
    assertEquals("The launch code is Orchid.", answer.content());
    assertEquals(
        "https://example.com/launch",
        answer.sources().getFirst().textSegment().metadata().getString("source"));
    assertEquals(1, server.getRequestCount());
  }

  @Test
  void frameworkOffloadRunsBlockingSdkRetrievalFromAsyncRag() throws Exception {
    events(
        chunk(CHUNK, MEMORY, "Async workflow evidence", 0.7), definition(MEMORY, SPACE, Map.of()));
    var augmentor =
        DefaultRetrievalAugmentor.builder()
            .contentRetriever(retriever().build())
            .offloadBlocking(true)
            .build();
    var question = UserMessage.from("question");
    var metadata = dev.langchain4j.rag.query.Metadata.from(question, null, List.of(question));
    var result =
        augmentor
            .augmentAsync(new AugmentationRequest(question, metadata))
            .get(3, TimeUnit.SECONDS);
    assertEquals("Async workflow evidence", result.contents().getFirst().textSegment().text());
    assertEquals(1, server.getRequestCount());
  }

  @Test
  void standardListenersSeeSuccessfulAndFailedRetrieval() {
    List<String> callbacks = new ArrayList<>();
    var observable =
        retriever()
            .build()
            .addListener(
                new ContentRetrieverListener() {
                  @Override
                  public void onRequest(ContentRetrieverRequestContext context) {
                    callbacks.add("request");
                  }

                  @Override
                  public void onResponse(ContentRetrieverResponseContext context) {
                    callbacks.add("response");
                  }

                  @Override
                  public void onError(ContentRetrieverErrorContext context) {
                    callbacks.add("error");
                  }
                });
    events("{}");
    observable.retrieve(Query.from("empty"));
    events(status("EMBEDDER_FAILED"));
    assertThrows(GoodMemRetrievalException.class, () -> observable.retrieve(Query.from("failure")));
    assertEquals(List.of("request", "response", "request", "error"), callbacks);
  }
}
