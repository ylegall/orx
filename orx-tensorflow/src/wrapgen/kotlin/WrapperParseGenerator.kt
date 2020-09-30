import com.github.javaparser.utils.SourceRoot
import org.tensorflow.Operand
import org.tensorflow.types.TInt32
import java.net.URI
import java.nio.file.Path


fun main() {
    val sourceRoot = SourceRoot(Path.of(URI("file:/home/rndr/git/tensorflow/java/tensorflow-core/tensorflow-core-api/src/gen/annotations")))
    sourceRoot.tryToParse()

    val sourceRoot2 = SourceRoot(Path.of(URI("file:/home/rndr/git/tensorflow/java/tensorflow-core/tensorflow-core-api/src/main/java/org/tensorflow")))
    sourceRoot2.tryToParse()


    val opsRoot = SourceRoot(Path.of(URI("file:/home/rndr/git/tensorflow/java/tensorflow-core/tensorflow-core-api/src/gen/java/org/tensorflow/op")))
    opsRoot.tryToParse()




    val opsGroup = "Ops"
    val opsVal = opsGroup.take(1).toLowerCase() + opsGroup.drop(1)

    val opsUnit = sourceRoot.parse("org.tensorflow.op", "$opsGroup.java")

    val opsClass = opsUnit.primaryType.get()

    println("interface K$opsGroup {")

    println("\tval $opsVal: $opsGroup")

    opsClass.methods.forEach { method ->

        if (method.type.isClassOrInterfaceType) {

            val returnType = method.type.asClassOrInterfaceType().name.asString()
            val parameters = method.parameters.joinToString(", ") {
                if (!it.isVarArgs) {
                    "${it.name}: ${it.type}"
                } else {
                    "varargs ${it.name}: ${it.type}"
                }
            }
            val parameterPass = method.parameters.joinToString(", ") {
                if (!it.isVarArgs) {
                    "${it.name}"
                } else {
                    "*${it.name}"
                }
            }

            val genericType = ("<" + method.typeParameters.joinToString(", ") {
                it.toString().replace("extends", ":")
            } + ">").replace("<>","")

            val opCU = opsRoot.compilationUnits.find {
                it.primaryTypeName.get() == returnType
            } ?: sourceRoot2.compilationUnits.find {
                it.primaryTypeName.get() == returnType
            }
            if (opCU != null) {
                val opClass = opCU.primaryType.get().asClassOrInterfaceDeclaration()
                val outputMethod = opClass.getMethodsByName("asOutput").firstOrNull()

                if (outputMethod != null) {
                    println("""
fun $genericType ${method.name}($parameters) : ${outputMethod.type} { 
    val op = $opsVal.${method.name}($parameterPass)
    return op.asOutput()
}""".split("\n").joinToString("\n") { "\t$it" })
                }
            } else {
                println("// omitted ${method.name}:$returnType")
            }

        } else {
            println("// omitted ${method.nameAsString}")
        }
    }
    println("}")
}