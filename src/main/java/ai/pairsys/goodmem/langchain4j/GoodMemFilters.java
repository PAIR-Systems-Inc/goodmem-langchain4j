package ai.pairsys.goodmem.langchain4j;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Collectors;

/** Safe construction of common native GoodMem metadata expressions. */
public final class GoodMemFilters {
  private GoodMemFilters() {}

  /**
   * Matches a top-level metadata field against a text value, escaping the value as data.
   *
   * @param key field name containing letters, digits and underscores, starting with a letter or
   *     underscore
   * @param value text to match, including quotes, backslashes or line breaks
   * @return a native GoodMem filter expression
   * @throws IllegalArgumentException if the key is not a simple field name or the value contains
   *     NUL
   */
  public static String textEquals(String key, String value) {
    Objects.requireNonNull(key, "key");
    Objects.requireNonNull(value, "value");
    if (!key.matches("[a-zA-Z_][a-zA-Z0-9_]*")) {
      throw new IllegalArgumentException("key must be a simple top-level metadata field name");
    }
    if (value.indexOf('\0') >= 0) {
      throw new IllegalArgumentException("Filter text cannot contain NUL");
    }
    // Isolate backslashes so server escape decoding cannot reinterpret a literal backslash+n.
    String literal =
        Arrays.stream(value.split("\\\\", -1))
            .map(GoodMemFilters::quote)
            .collect(Collectors.joining(" || '\\\\' || "));
    return "CAST(val('$." + key + "') AS TEXT) = (" + literal + ")";
  }

  private static String quote(String value) {
    return "'"
        + value.replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        + "'";
  }
}
