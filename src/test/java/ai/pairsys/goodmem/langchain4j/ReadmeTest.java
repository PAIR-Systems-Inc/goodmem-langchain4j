package ai.pairsys.goodmem.langchain4j;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import javax.tools.ToolProvider;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ReadmeTest {
  @TempDir Path directory;

  @Test
  void quickstartCompilesAgainstThePublicApi() throws Exception {
    String readme = Files.readString(Path.of("README.md"));
    var example = Pattern.compile("```java\\R(.*?)```", Pattern.DOTALL).matcher(readme);
    assertTrue(example.find(), "README must contain its runnable quickstart");
    compile("Quickstart", example.group(1));
    assertFalse(example.find(), "Keep the README focused on one runnable quickstart");
  }

  @Test
  void usageGuideExamplesCompileWithTheirDocumentedContext() throws Exception {
    var examples =
        Pattern.compile("```java\\R(.*?)```", Pattern.DOTALL)
            .matcher(Files.readString(Path.of("docs/usage.md")));
    List<String> imports =
        new ArrayList<>(
            List.of(
                "import ai.pairsys.goodmem.client.Goodmem;",
                "import ai.pairsys.goodmem.langchain4j.GoodMemContentRetriever;",
                "import dev.langchain4j.model.chat.ChatModel;",
                "import dev.langchain4j.rag.DefaultRetrievalAugmentor;",
                "import java.util.List;"));
    StringBuilder methods = new StringBuilder();
    int index = 0;
    while (examples.find()) {
      methods.append("void example").append(index++).append("() {\n");
      for (String line : examples.group(1).lines().toList()) {
        if (line.startsWith("import ")) {
          imports.add(line);
        } else {
          methods.append(line).append('\n');
        }
      }
      methods.append("}\n");
    }
    assertTrue(index > 0, "Usage guide must contain examples");
    compile(
        "GuideExamples",
        String.join("\n", imports)
            + "\npublic class GuideExamples {\n"
            + "Goodmem client; GoodMemContentRetriever retriever; ChatModel model; String spaceId, rerankerId;\n"
            + methods
            + "}\n");
  }

  private void compile(String className, String code) throws Exception {
    Path source = directory.resolve(className + ".java");
    Files.writeString(source, code);
    var compiler = ToolProvider.getSystemJavaCompiler();
    assertNotNull(compiler, "Run the tests with a JDK");
    var diagnostics = new StringWriter();
    try (var files = compiler.getStandardFileManager(null, null, null)) {
      boolean compiled =
          compiler
              .getTask(
                  diagnostics,
                  files,
                  null,
                  List.of(
                      "--release",
                      "21",
                      "-proc:none",
                      "-classpath",
                      System.getProperty("java.class.path"),
                      "-d",
                      directory.toString()),
                  null,
                  files.getJavaFileObjects(source))
              .call();
      assertTrue(compiled, diagnostics.toString());
    }
  }
}
