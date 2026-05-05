package ai.pairsys.langchain4j.goodmem;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.AiServices;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

/**
 * End-to-end conversation test: a real OpenAI-powered agent uses GoodMem tools
 * to store and retrieve memories, just like a real user would interact with it.
 */
@EnabledIfEnvironmentVariable(named = "OPENAI_API_KEY", matches = ".+")
@EnabledIfEnvironmentVariable(named = "GOODMEM_API_KEY", matches = ".+")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class GoodMemConversationIT {

    interface Assistant {
        String chat(String userMessage);
    }

    private static Assistant assistant;

    @BeforeAll
    static void setUp() {
        String openAiKey = System.getenv("OPENAI_API_KEY");
        String goodMemUrl = System.getenv("GOODMEM_BASE_URL");
        String goodMemKey = System.getenv("GOODMEM_API_KEY");

        if (goodMemUrl == null || goodMemUrl.isEmpty()) {
            goodMemUrl = "http://localhost:8080";
        }

        OpenAiChatModel model = OpenAiChatModel.builder()
                .apiKey(openAiKey)
                .modelName("gpt-4o-mini")
                .temperature(0.0)
                .build();

        GoodMemTools goodmem = GoodMemTools.builder()
                .baseUrl(goodMemUrl)
                .apiKey(goodMemKey)
                .verifySsl(false)
                .build();

        assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(30))
                .tools(goodmem)
                .build();
    }

    @Test
    @Order(1)
    void step1_listEmbeddersAndCreateSpace() {
        System.out.println("\n══════════════════════════════════════════");
        System.out.println("STEP 1: Ask the agent to set up a space");
        System.out.println("══════════════════════════════════════════");

        String response = assistant.chat(
                "List the available embedders and then create a space called "
                + "'demo-convo-" + System.currentTimeMillis() + "' "
                + "using a non-OpenAI embedder if available. "
                + "Tell me the space ID when done.");

        System.out.println("\nUser: List embedders and create a space for me.");
        System.out.println("\nAgent: " + response);

        // We don't assert exact text — the LLM's response is non-deterministic.
        // But it should mention a space ID (UUID pattern).
        assert response != null && !response.isEmpty() : "Agent should respond";
    }

    @Test
    @Order(2)
    void step2_storeMemories() {
        System.out.println("\n══════════════════════════════════════════");
        System.out.println("STEP 2: Store some facts as memories");
        System.out.println("══════════════════════════════════════════");

        String r1 = assistant.chat(
                "Store this as a memory in the space you just created: "
                + "'Acme Corp was founded in 2019 by Jane Smith in Austin, Texas. "
                + "The company builds AI-powered developer tools.'");
        System.out.println("\nUser: Store a fact about Acme Corp.");
        System.out.println("Agent: " + r1);

        String r2 = assistant.chat(
                "Also store this memory: "
                + "'Acme Corp raised a $50 million Series B in March 2024, "
                + "led by Sequoia Capital. The company has 150 employees.'");
        System.out.println("\nUser: Store another fact about funding.");
        System.out.println("Agent: " + r2);

        String r3 = assistant.chat(
                "Store one more: "
                + "'Acme Corp\\'s flagship product is CodePilot, an AI code review tool "
                + "that integrates with GitHub and GitLab. It costs $29/month per seat.'");
        System.out.println("\nUser: Store a fact about the product.");
        System.out.println("Agent: " + r3);
    }

    @Test
    @Order(3)
    void step3_askQuestionsFromMemory() {
        System.out.println("\n══════════════════════════════════════════");
        System.out.println("STEP 3: Ask questions — agent should search memories");
        System.out.println("══════════════════════════════════════════");

        String q1 = assistant.chat("Who founded Acme Corp and where is it based?");
        System.out.println("\nUser: Who founded Acme Corp and where is it based?");
        System.out.println("Agent: " + q1);

        String q2 = assistant.chat("How much funding has Acme Corp raised and who led the round?");
        System.out.println("\nUser: How much funding has Acme Corp raised?");
        System.out.println("Agent: " + q2);

        String q3 = assistant.chat("What does CodePilot cost and what platforms does it support?");
        System.out.println("\nUser: What does CodePilot cost?");
        System.out.println("Agent: " + q3);
    }

    @Test
    @Order(4)
    void step4_cleanup() {
        System.out.println("\n══════════════════════════════════════════");
        System.out.println("STEP 4: Ask agent to clean up");
        System.out.println("══════════════════════════════════════════");

        String response = assistant.chat(
                "Delete all the memories you stored in this conversation. "
                + "List each memory ID you delete.");
        System.out.println("\nUser: Delete all the memories you stored.");
        System.out.println("Agent: " + response);
    }
}
