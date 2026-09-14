package ai.pairsys.goodmem.langchain4j;

import ai.pairsys.goodmem.client.Goodmem;
import ai.pairsys.goodmem.client.models.ChunkId;
import ai.pairsys.goodmem.client.models.GoodMemStatus;
import ai.pairsys.goodmem.client.models.GoodMemStatusCode;
import ai.pairsys.goodmem.client.models.Memory;
import ai.pairsys.goodmem.client.models.MemoryId;
import ai.pairsys.goodmem.client.models.RetrieveMemoryEvent;
import ai.pairsys.goodmem.client.models.RetrieveMemoryRequest;
import ai.pairsys.goodmem.client.models.SpaceId;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.ContentMetadata;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;

/** Converts SDK events to framework content; HTTP and stream parsing belong to the SDK. */
final class RetrievalResults {
  private static final ObjectMapper JSON = new ObjectMapper();

  private RetrievalResults() {}

  static List<RetrieveMemoryEvent> read(Goodmem client, RetrieveMemoryRequest request) {
    try (var stream = client.memories.retrieve(request)) {
      List<RetrieveMemoryEvent> events = new ArrayList<>();
      stream.forEach(events::add);
      return List.copyOf(events);
    }
  }

  static List<Content> content(
      List<RetrieveMemoryEvent> events, Set<SpaceId> spaces, int limit, boolean reranked) {
    List<GoodMemStatus> problems =
        events.stream()
            .map(RetrieveMemoryEvent::status)
            .filter(Objects::nonNull)
            .filter(s -> s.code() != GoodMemStatusCode.UNKNOWN && !informational(s))
            .toList();
    if (!problems.isEmpty()) {
      throw new GoodMemRetrievalException(problems);
    }
    Map<MemoryId, Memory> memories = memories(events);
    Set<ChunkId> seen = new HashSet<>();
    List<Content> results = new ArrayList<>();
    for (var event : events) {
      if (event.retrievedItem() == null || event.retrievedItem().chunk() == null) {
        continue;
      }
      var reference = event.retrievedItem().chunk();
      var chunk = reference.chunk();
      if (chunk == null
          || chunk.chunkId() == null
          || chunk.memoryId() == null
          || chunk.chunkText() == null
          || chunk.chunkText().isBlank()) {
        throw new GoodMemException("Retrieval returned a chunk without text or identifiers");
      }
      Memory memory = memories.get(chunk.memoryId());
      if (memory == null) {
        throw new GoodMemException("Retrieval omitted metadata for memory " + chunk.memoryId());
      }
      if (!spaces.contains(memory.spaceId())) {
        throw new GoodMemException("Retrieval returned a memory outside the configured spaces");
      }
      if (!seen.add(chunk.chunkId())) {
        continue;
      }
      Map<String, Object> metadata = new LinkedHashMap<>();
      copyMetadata(metadata, memory.metadata());
      copyMetadata(metadata, chunk.metadata());
      var memoryMetadata = memory.metadata() == null ? Map.<String, Object>of() : memory.metadata();
      var chunkMetadata = chunk.metadata() == null ? Map.<String, Object>of() : chunk.metadata();
      Set<String> reserved = Set.of("memory_id", "chunk_id", "space_id", "goodmem_metadata");
      boolean collision =
          memoryMetadata.keySet().stream().anyMatch(chunkMetadata::containsKey)
              || memoryMetadata.keySet().stream().anyMatch(reserved::contains)
              || chunkMetadata.keySet().stream().anyMatch(reserved::contains);
      if (collision) {
        // LangChain4j Metadata accepts scalar values, so keep the originals as JSON.
        copyMetadata(
            metadata,
            Map.of("goodmem_metadata", Map.of("memory", memoryMetadata, "chunk", chunkMetadata)));
      }
      if (!metadata.containsKey("source") && memory.originalContentRef() != null) {
        metadata.put("source", memory.originalContentRef());
      }
      metadata.put("memory_id", chunk.memoryId().toString());
      metadata.put("chunk_id", chunk.chunkId().toString());
      metadata.put("space_id", memory.spaceId().toString());
      Map<ContentMetadata, Object> scores = new EnumMap<>(ContentMetadata.class);
      if (reference.relevanceScore() != null) {
        scores.put(ContentMetadata.SCORE, reference.relevanceScore());
        if (reranked) {
          scores.put(ContentMetadata.RERANKED_SCORE, reference.relevanceScore());
        }
      }
      if (results.size() < limit) {
        results.add(
            Content.from(TextSegment.from(chunk.chunkText(), Metadata.from(metadata)), scores));
      }
    }
    return List.copyOf(results);
  }

  private static Map<MemoryId, Memory> memories(List<RetrieveMemoryEvent> events) {
    Map<MemoryId, Memory> memories = new HashMap<>();
    for (var event : events) {
      Memory memory = event.memoryDefinition();
      if (memory == null && event.retrievedItem() != null) {
        memory = event.retrievedItem().memory();
      }
      if (memory != null) {
        memories.put(memory.memoryId(), memory);
      }
    }
    return memories;
  }

  static GoodMemTools.RetrievalResult compact(List<RetrieveMemoryEvent> events, int limit) {
    var statuses =
        events.stream().map(RetrieveMemoryEvent::status).filter(Objects::nonNull).toList();
    boolean partial = statuses.stream().anyMatch(s -> !informational(s));
    var memories = memories(events);
    var chunks = new ArrayList<GoodMemTools.RetrievedChunk>();
    Set<ChunkId> seen = new HashSet<>();
    StringBuilder reply = new StringBuilder();
    for (var event : events) {
      if (event.abstractReply() != null && event.abstractReply().text() != null) {
        reply.append(event.abstractReply().text());
      }
      if (event.retrievedItem() == null || event.retrievedItem().chunk() == null) {
        continue;
      }
      var reference = event.retrievedItem().chunk();
      var chunk = reference.chunk();
      if (chunk == null
          || chunk.chunkId() == null
          || chunk.memoryId() == null
          || chunk.chunkText() == null
          || chunk.chunkText().isBlank()) {
        partial = true;
        continue;
      }
      Memory memory = memories.get(chunk.memoryId());
      if (memory == null) {
        partial = true;
      }
      if (seen.add(chunk.chunkId()) && chunks.size() < limit) {
        Object source = chunk.metadata() == null ? null : chunk.metadata().get("source");
        if (!(source instanceof String) && memory != null) {
          source = memory.metadata() == null ? null : memory.metadata().get("source");
          if (!(source instanceof String)) {
            source = memory.originalContentRef();
          }
        }
        chunks.add(
            new GoodMemTools.RetrievedChunk(
                chunk.chunkText(),
                source instanceof String s ? s : null,
                reference.relevanceScore(),
                chunk.memoryId(),
                chunk.chunkId(),
                memory == null ? null : memory.spaceId()));
      }
    }
    return new GoodMemTools.RetrievalResult(
        List.copyOf(chunks), reply.isEmpty() ? null : reply.toString(), statuses, partial);
  }

  private static boolean informational(GoodMemStatus status) {
    return status.code() == GoodMemStatusCode.LLM_CAPABILITY_INFERRED
        || (status.code() == GoodMemStatusCode.FEATURE_DISABLED
            && status.details() != null
            && "summarization".equals(status.details().get("feature"))
            && "llm_id".equals(status.details().get("required_param")));
  }

  private static void copyMetadata(Map<String, Object> target, Map<String, Object> source) {
    if (source == null) {
      return;
    }
    source.forEach(
        (key, value) -> {
          if (value == null) {
            return;
          }
          if (value instanceof String
              || value instanceof Integer
              || value instanceof Long
              || value instanceof Float
              || value instanceof Double
              || value instanceof UUID) {
            target.put(key, value);
          } else {
            try {
              target.put(key, JSON.writeValueAsString(value));
            } catch (JsonProcessingException e) {
              throw new GoodMemException("Cannot represent metadata field " + key, e);
            }
          }
        });
  }
}
