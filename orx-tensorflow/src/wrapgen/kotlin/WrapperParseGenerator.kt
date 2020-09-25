import com.github.javaparser.utils.SourceRoot
import java.net.URI
import java.nio.file.Path


fun main() {
    val sourceRoot = SourceRoot(Path.of(URI("file:/home/rndr/git/tensorflow/java/tensorflow-core/tensorflow-core-api/src/gen/annotations")))
    sourceRoot.tryToParse()

    val opsRoot = SourceRoot(Path.of(URI("file:/home/rndr/git/tensorflow/java/tensorflow-core/tensorflow-core-api/src/gen/java/org/tensorflow/op")))
    opsRoot.tryToParse()

    val opsGroup = "MathOps"
    val opsUnit = sourceRoot.parse("org.tensorflow.op", "$opsGroup.java")

    val opsClass = opsUnit.primaryType.get()

    println("interface $opsGroup(val scope:Scope) {")

    opsClass.methods.forEach { method ->
        val returnType = method.type.asClassOrInterfaceType().name.asString()
        val parameters = method.parameters.joinToString(", ") {
            "${it.name} : ${it.type}"
        }
        val parameterPass = method.parameters.joinToString(", ") {
            "${it.name}"
        }

        val genericType = method.typeParameters.joinToString(", ") {
            it.toString().replace("extends", ":")
        }

        val opCU = opsRoot.compilationUnits.find {
            it.primaryTypeName.get() == returnType
        }!!
        val opClass = opCU.primaryType.get().asClassOrInterfaceDeclaration()
        val outputMethod = opClass.getMethodsByName("asOutput").firstOrNull()

        if (outputMethod != null) {
            println("""
fun <$genericType> ${method.name}($parameters) : ${outputMethod.type} { 
    val op = $opsGroup.${method.name}(scope, $parameterPass)
    return op.asOutput()
}""".split("\n").joinToString("\n") { "\t$it"})
        }

    }
    println("}")
}