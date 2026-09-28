package ai.pairsys.goodmem.langchain4j;

import java.util.Locale;
import java.util.regex.Pattern;

/**
 * The integration's single ID guard. The SDK places IDs in URL paths and OkHttp resolves dot
 * segments, so {@code ../spaces/<id>} given as a memory ID would address a space. Every externally
 * supplied GoodMem ID passes through here before any request is made.
 */
final class GoodMemIds {
  private static final Pattern UUID =
      Pattern.compile(
          "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}");

  private GoodMemIds() {}

  /**
   * Returns the ID as a lowercase canonical UUID, or refuses it.
   *
   * @param value ID from a model, developer or configuration
   * @param field argument name reported in the error
   * @return the lowercase UUID
   * @throws IllegalArgumentException if the value is not a canonical UUID
   */
  static String requireUuid(String value, String field) {
    if (value == null || !UUID.matcher(value).matches()) {
      throw new IllegalArgumentException(
          field + " must be a UUID, such as 123e4567-e89b-12d3-a456-426614174000");
    }
    return value.toLowerCase(Locale.ROOT);
  }
}
