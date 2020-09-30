import org.openrndr.extra.tensorflow.graph
import org.openrndr.extra.tensorflow.summary
import org.openrndr.extra.tensorflow.toFloatArray
import org.tensorflow.Session
import org.tensorflow.types.TFloat32

fun main() {
    val g = graph {
        val a = array(10)
        val b = array(10)
        val c = a + b

        val d = cos(cast(a, TFloat32.DTYPE))

        val s = sigmoid(d)

        println(d.op())
    }
    val s = Session(g)
    val runner = s.runner()
    runner.fetch("Cos")
    val result = runner.run()
    result[0].summary()
    println(result[0].expect(TFloat32.DTYPE).toFloatArray()[0])
}