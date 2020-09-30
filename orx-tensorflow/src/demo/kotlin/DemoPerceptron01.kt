
import org.openrndr.extra.tensorflow.graph
import org.tensorflow.Output
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.train.ApplyAdam
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32
import org.tensorflow.types.family.TNumber

fun main() {
    val g = graph {
        val x = floatPlaceholder(shape = Shape.of(1, 784))
        val y = floatPlaceholder(shape = Shape.of(1, 10))

        val w = variable(zeros(shape(784, 10), TFloat32.DTYPE))
        val b = variable(zeros(shape(10), TFloat32.DTYPE))

        val activation = softmax(matMul(x, w) + b)
        val crossEntropy = y * log(activation)


        val cost = reduceSum( -reduceSum(crossEntropy, constant(1)), constant(-1))
        val bla = ops.variable(ops.zeros(shape(784, 10), TFloat32.DTYPE))
        log(bla)
    }



}