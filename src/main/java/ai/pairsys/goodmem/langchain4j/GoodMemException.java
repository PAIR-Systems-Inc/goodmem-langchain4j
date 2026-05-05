package ai.pairsys.goodmem.langchain4j;

/**
 * Exception thrown when a GoodMem API operation fails.
 * <p>
 * Wraps HTTP errors, connection failures, and other GoodMem-specific problems
 * with descriptive messages to help the user understand what went wrong.
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
     * @param cause   underlying cause of the failure
     */
    public GoodMemException(String message, Throwable cause) {
        super(message, cause);
    }
}
