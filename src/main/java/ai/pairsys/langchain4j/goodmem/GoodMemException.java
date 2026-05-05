package ai.pairsys.langchain4j.goodmem;

/**
 * Exception thrown when a GoodMem API operation fails.
 * <p>
 * Wraps HTTP errors, connection failures, and other GoodMem-specific problems
 * with descriptive messages to help the user understand what went wrong.
 */
public class GoodMemException extends RuntimeException {

    public GoodMemException(String message) {
        super(message);
    }

    public GoodMemException(String message, Throwable cause) {
        super(message, cause);
    }
}
