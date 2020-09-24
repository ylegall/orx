import org.tensorflow.Graph
import org.tensorflow.Session

fun main() {

    val g = Graph()
    println(g)
    val s = Session(g)
    println(s)


}