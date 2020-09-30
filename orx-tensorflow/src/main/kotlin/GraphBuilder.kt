package org.openrndr.extra.tensorflow

import org.openrndr.extra.tensorflow.ops.*
import org.tensorflow.*
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.*
import org.tensorflow.op.core.*
import org.tensorflow.op.linalg.MatMul
import org.tensorflow.op.math.*
import org.tensorflow.types.*
import org.tensorflow.types.family.TType

class GraphBuilder(executionEnvironment: ExecutionEnvironment)
    : KAudioOps,
        KBitwiseOps,
        KDataOps,
        KDtypesOps,
        KImageOps,
        KIoOps,
        KLinalgOps,
        KMathOps,
        KNnOps,
        KOps,
        KQuantizationOps,
        KRaggedOps,
        KRandomOps,
        KShapeOps,
        KSignalOps,
        KSparseOps,
        KStringsOps,
        KSummaryOps,
        KTrainOps,
        KXlaOps {

    override val ops = Ops.create(executionEnvironment)
    override val audioOps by lazy { ops.audio }
    override val bitwiseOps by lazy { ops.bitwise }
    override val dataOps by lazy { ops.data }
    override val dtypesOps by lazy { ops.dtypes }
    override val imageOps by lazy { ops.image }
    override val ioOps by lazy { ops.io }
    override val linalgOps by lazy { ops.linalg }
    override val mathOps by lazy { ops.math }
    override val nnOps by lazy { ops.nn }
    override val quantizationOps by lazy { ops.quantization }
    override val raggedOps by lazy { ops.ragged }
    override val randomOps by lazy { ops.random }
    override val shapeOps by lazy { ops.shape }
    override val signalOps by lazy { ops.signal }
    override val sparseOps by lazy { ops.sparse }
    override val stringsOps by lazy { ops.strings }
    override val summaryOps by lazy { ops.summary }
    override val trainOps by lazy { ops.train }
    override val xlaOps by lazy { ops.xla }

    val scope by lazy { ops.scope() }
    val inputs = mutableMapOf<String, Tensor<*>?>()
    val outputs = mutableMapOf<String, Tensor<*>?>()

    fun output(out: Output<*>) {
        outputs[out.op().name()] = null
    }

    fun floatPlaceholder(value: Tensor<TFloat32>? = null, shape: Shape? = null): Output<TFloat32> {

        val ph = if (shape != null) Placeholder.create(scope, TFloat32.DTYPE, Placeholder.shape(shape)) else {
            Placeholder.create(scope, TFloat32.DTYPE)
        }
        val out = ph.asOutput() as Output<TFloat32>
        inputs[out.op().name()] = value

        return out
    }

    fun <T : TType> Output<T>.matMul(y: Output<T>): Output<T> {
        return MatMul.create(scope, this, y).asOutput()
    }

    operator fun <T : TType> Output<T>.times(scale: Output<T>): Output<T> {
        return Mul.create(scope, this, scale).asOutput()
    }

    operator fun <T : TType> Output<T>.plus(add: Output<T>): Output<T> {
        return Add.create(scope, this, add).op().output(0)
    }

    operator fun <T : TType> Output<T>.minus(add: Output<T>): Output<T> {
        return Sub.create(scope, this, add).op().output(0)
    }

    operator fun <T : TType> Output<T>.unaryMinus(): Output<T> {
        return Neg.create(scope, this).op().output(0)
    }


}

fun graph(builder: GraphBuilder.() -> Unit): Graph {
    val graph = Graph()
    val gb = GraphBuilder(graph)
    gb.builder()
    return graph
}