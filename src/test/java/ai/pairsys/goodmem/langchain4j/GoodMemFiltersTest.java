package ai.pairsys.goodmem.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class GoodMemFiltersTest {
  @Test
  void quotesUntrustedTextUsingGoodmemGrammar() {
    assertEquals(
        "CAST(val('$.team') AS TEXT) = ('O\\'Reilly')",
        GoodMemFilters.textEquals("team", "O'Reilly"));
    assertEquals(
        "CAST(val('$.team') AS TEXT) = ('a' || '\\\\' || 'notes\\n')",
        GoodMemFilters.textEquals("team", "a\\notes\n"));
    assertEquals(
        "CAST(val('$.team') AS TEXT) = ('\\' OR TRUE OR \\'x\\' = \\'x')",
        GoodMemFilters.textEquals("team", "' OR TRUE OR 'x' = 'x"));
    assertEquals("CAST(val('$._id2') AS TEXT) = ('')", GoodMemFilters.textEquals("_id2", ""));
  }

  @Test
  void rejectsKeysThatCouldChangeTheExpressionAndUnsupportedNul() {
    assertThrows(
        IllegalArgumentException.class,
        () -> GoodMemFilters.textEquals("team') OR TRUE", "support"));
    assertThrows(
        IllegalArgumentException.class, () -> GoodMemFilters.textEquals("$.team", "support"));
    assertThrows(
        IllegalArgumentException.class, () -> GoodMemFilters.textEquals("team", "bad\0value"));
  }
}
