package ai.pairsys.goodmem.langchain4j;

import ai.pairsys.goodmem.client.Goodmem;
import ai.pairsys.goodmem.client.models.ChunkId;
import ai.pairsys.goodmem.client.models.ChunkReference;
import ai.pairsys.goodmem.client.models.GoodMemStatus;
import ai.pairsys.goodmem.client.models.GoodMemStatusCode;
import ai.pairsys.goodmem.client.models.Memory;
import ai.pairsys.goodmem.client.models.MemoryId;
import ai.pairsys.goodmem.client.models.ResultSetBoundary;
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
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Converts SDK events to framework content; HTTP and stream parsing belong to the SDK. */
final class RetrievalResults {
  private static final ObjectMapper JSON = new ObjectMapper();
  private static final Logger LOG = LoggerFactory.getLogger(RetrievalResults.class);
  static final String PARTIAL = "goodmem_partial";
  static final String STATUSES = "goodmem_statuses";
  static final String SCORE_TYPE = "goodmem_score_type";
  private static final Set<String> RESERVED =
      Set.of(
          "memory_id", "chunk_id", "space_id", "goodmem_metadata", PARTIAL, STATUSES, SCORE_TYPE);

  private RetrievalResults() {}

  static List<RetrieveMemoryEvent> read(Goodmem client, RetrieveMemoryRequest request) {
    try (var stream = client.memories.retrieve(request)) {
      List<RetrieveMemoryEvent> events = new ArrayList<>();
      stream.forEach(events::add);
      return List.copyOf(events);
    }
  }

  /**
   * Contents for the framework plus the non-informational diagnostics they were retrieved with.
   * {@code vectorFallback} means a configured reranker did not run; {@code textlessItems} counts
   * retrieved items skipped because they had no text or identifiers, and {@code undefinedMemories}
   * counts kept hits whose memory definition the stream did not carry.
   */
  record ContentResult(
      List<Content> contents,
      List<GoodMemStatus> statuses,
      boolean vectorFallback,
      int textlessItems,
      int undefinedMemories) {
    boolean partial() {
      return !statuses.isEmpty() || textlessItems > 0 || undefinedMemories > 0;
    }
  }

  private record Hit(ChunkReference reference, Memory memory) {}

  static ContentResult content(
      List<RetrieveMemoryEvent> events,
      Set<SpaceId> spaces,
      int limit,
      boolean reranked,
      boolean failOnIncomplete) {
    List<GoodMemStatus> problems =
        events.stream()
            .map(RetrieveMemoryEvent::status)
            .filter(Objects::nonNull)
            .filter(s -> !informational(s))
            .toList();
    if (failOnIncomplete) {
      // Opt-in legacy behaviour: known codes abort; future (UNKNOWN) codes never did.
      var known = problems.stream().filter(s -> s.code() != GoodMemStatusCode.UNKNOWN).toList();
      if (!known.isEmpty()) {
        throw new GoodMemRetrievalException(known);
      }
    }
    boolean rerankerScored = reranked && rerankerScored(events, problems);
    Map<MemoryId, Memory> memories = memories(events);
    Set<ChunkId> seen = new HashSet<>();
    List<Hit> hits = new ArrayList<>();
    int textless = 0;
    int undefined = 0;
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
        if (failOnIncomplete) {
          throw new GoodMemException("Retrieval returned a chunk without text or identifiers");
        }
        // Nothing to hand the framework; the other hits are still valid (contract Q4a).
        textless++;
        continue;
      }
      Memory memory = memories.get(chunk.memoryId());
      if (memory == null) {
        if (failOnIncomplete) {
          throw new GoodMemException("Retrieval omitted metadata for memory " + chunk.memoryId());
        }
        // The server could not load this memory (e.g. MEMORY_LOAD_FAILED); keep its text.
        undefined++;
      } else if (!spaces.contains(memory.spaceId())) {
        throw new GoodMemException("Retrieval returned a memory outside the configured spaces");
      }
      if (seen.add(chunk.chunkId()) && hits.size() < limit) {
        hits.add(new Hit(reference, memory));
      }
    }
    boolean partial = !problems.isEmpty() || textless > 0 || undefined > 0;
    String statusJson = partial ? statusJson(problems) : null;
    List<Content> results = new ArrayList<>();
    for (var hit : hits) {
      results.add(content(hit, statusJson, rerankerScored));
    }
    if (partial) {
      LOG.warn(
          "GoodMem retrieval was incomplete; returning {} result(s) marked {}=true{}{}{}: {}",
          results.size(),
          PARTIAL,
          reranked && !rerankerScored ? " in vector order because reranking did not run" : "",
          textless > 0 ? "; skipped " + textless + " item(s) without text or identifiers" : "",
          undefined > 0 ? "; kept " + undefined + " item(s) without memory metadata" : "",
          problems.isEmpty() ? "no status reported" : summary(problems));
    }
    return new ContentResult(
        List.copyOf(results), problems, reranked && !rerankerScored, textless, undefined);
  }

  private static Content content(Hit hit, String statusJson, boolean rerankerScored) {
    var reference = hit.reference();
    var chunk = reference.chunk();
    Memory memory = hit.memory();
    Map<String, Object> memoryMetadata =
        memory == null || memory.metadata() == null ? Map.of() : memory.metadata();
    Map<String, Object> chunkMetadata = chunk.metadata() == null ? Map.of() : chunk.metadata();
    Map<String, Object> metadata = new LinkedHashMap<>();
    copyMetadata(metadata, memoryMetadata);
    copyMetadata(metadata, chunkMetadata);
    boolean collision =
        memoryMetadata.keySet().stream().anyMatch(chunkMetadata::containsKey)
            || memoryMetadata.keySet().stream().anyMatch(RESERVED::contains)
            || chunkMetadata.keySet().stream().anyMatch(RESERVED::contains);
    if (collision) {
      // LangChain4j Metadata accepts scalar values, so keep the originals as JSON.
      copyMetadata(
          metadata,
          Map.of("goodmem_metadata", Map.of("memory", memoryMetadata, "chunk", chunkMetadata)));
    }
    if (!metadata.containsKey("source") && memory != null && memory.originalContentRef() != null) {
      metadata.put("source", memory.originalContentRef());
    }
    metadata.put("memory_id", chunk.memoryId().toString());
    metadata.put("chunk_id", chunk.chunkId().toString());
    // Without a definition the space is unknown; stored metadata must not stand in for it.
    metadata.remove("space_id");
    if (memory != null) {
      metadata.put("space_id", memory.spaceId().toString());
    }
    // Integration-owned markers are never taken from stored metadata (originals stay above).
    metadata.remove(PARTIAL);
    metadata.remove(STATUSES);
    metadata.remove(SCORE_TYPE);
    if (statusJson != null) {
      metadata.put(PARTIAL, "true");
      metadata.put(STATUSES, statusJson);
    }
    Map<ContentMetadata, Object> scores = new EnumMap<>(ContentMetadata.class);
    if (reference.relevanceScore() != null) {
      scores.put(ContentMetadata.SCORE, reference.relevanceScore());
      metadata.put(SCORE_TYPE, rerankerScored ? "reranker" : "vector");
      if (rerankerScored) {
        scores.put(ContentMetadata.RERANKED_SCORE, reference.relevanceScore());
      }
    }
    return Content.from(TextSegment.from(chunk.chunkText(), Metadata.from(metadata)), scores);
  }

  /**
   * Whether a configured reranker produced the scores. The stream names the stage that produced its
   * result set ({@code rerank} when reranking ran, {@code retrieve} for vector order), so that is
   * trusted when present. Older streams without stage names fall back to the statuses GoodMem sends
   * when the reranker is missing or fails.
   */
  private static boolean rerankerScored(
      List<RetrieveMemoryEvent> events, List<GoodMemStatus> problems) {
    List<String> stages =
        events.stream()
            .map(RetrieveMemoryEvent::resultSetBoundary)
            .filter(Objects::nonNull)
            .map(ResultSetBoundary::stageName)
            .filter(name -> name != null && !name.isBlank())
            .toList();
    if (!stages.isEmpty()) {
      return stages.stream().anyMatch("rerank"::equalsIgnoreCase);
    }
    return problems.stream()
        .noneMatch(
            s ->
                s.code() == GoodMemStatusCode.RERANKING_FAILED
                    || s.code() == GoodMemStatusCode.NOT_FOUND);
  }

  static String summary(List<GoodMemStatus> statuses) {
    return statuses.stream()
        .map(
            s -> s.message() == null ? s.code().name() : s.code().name() + " (" + s.message() + ")")
        .collect(Collectors.joining(", "));
  }

  static String statusJson(List<GoodMemStatus> statuses) {
    List<Map<String, Object>> list = new ArrayList<>();
    for (var status : statuses) {
      Map<String, Object> entry = new LinkedHashMap<>();
      entry.put("code", status.code().name());
      if (status.message() != null) {
        entry.put("message", status.message());
      }
      if (status.details() != null && !status.details().isEmpty()) {
        entry.put("details", status.details());
      }
      list.add(entry);
    }
    try {
      return JSON.writeValueAsString(list);
    } catch (JsonProcessingException e) {
      throw new GoodMemException("Cannot represent retrieval statuses", e);
    }
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

  /**
   * Contract Q1: these two codes are noise by code alone. {@code FEATURE_DISABLED} means the caller
   * did not configure an optional feature; a requested feature that failed has its own code. The
   * details are deliberately not inspected.
   */
  private static boolean informational(GoodMemStatus status) {
    return status.code() == GoodMemStatusCode.LLM_CAPABILITY_INFERRED
        || status.code() == GoodMemStatusCode.FEATURE_DISABLED;
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
