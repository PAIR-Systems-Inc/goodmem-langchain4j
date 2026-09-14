package ai.pairsys.goodmem.langchain4j;

/**
 * An indexing or response-contract failure in the LangChain4j integration. SDK request failures
 * propagate directly unless a write or readiness exception adds IDs needed for recovery; the
 * original SDK exception then remains available as the cause.
 */
public class GoodMemException extends RuntimeException {

  /**
   * Construct a new GoodMemException with the given detail message.
   *
   * @param message detail message describing the failure
   */
  public GoodMemException(String message) {
    super(message);
  }

  /**
   * Construct a new GoodMemException with the given detail message and underlying cause.
   *
   * @param message detail message describing the failure
   * @param cause underlying cause of the failure
   */
  public GoodMemException(String message, Throwable cause) {
    super(message, cause);
  }
}
