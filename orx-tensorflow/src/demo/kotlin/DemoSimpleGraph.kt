import org.openrndr.extra.tensorflow.summary
import org.openrndr.extra.tensorflow.toIntArray
import org.tensorflow.Session
import org.tensorflow.types.TInt32

fun main() {
    val g = graph {
        val a = arrayConstant(intArrayOf(10))
        val b = arrayConstant(intArrayOf(10))
        val c = a + b
        println(c.op())
    }
    val s = Session(g)
    val runner = s.runner()
    runner.fetch("Add")
    val result = runner.run()
    result[0].summary()
    println(result[0].expect(TInt32.DTYPE).toIntArray()[0])
}