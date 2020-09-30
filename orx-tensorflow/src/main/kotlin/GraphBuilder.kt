package org.openrndr.extra.tensorflow

import org.openrndr.extra.tensorflow.ops.*
import org.tensorflow.*
import org.tensorflow.op.*
import org.tensorflow.op.core.*
import org.tensorflow.op.dtypes.Cast
import org.tensorflow.op.image.CropAndResize
import org.tensorflow.op.image.ResizeBilinear
import org.tensorflow.op.linalg.MatMul
import org.tensorflow.op.math.*
import org.tensorflow.op.nn.*
import org.tensorflow.types.*
import org.tensorflow.types.family.TNumber
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
        KQuantizationOps,
        KRaggedOps,
        KRandomOps,
        KShapeOps,
        KSignalOps,
        KSparseOps,
        KStringsOps,
        KSummaryOps,
        KTrainOps,
        KXlaOps

        {

    val ops = Ops.create(executionEnvironment)
    override val audioOps by lazy { ops.audio }
    override val bitwiseOps by lazy { ops.bitwise }
    override val dataOps by lazy { ops.data }
    override val dtypesOps by lazy { ops.dtypes }
    override val imageOps by lazy { ops.image}
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

    fun floatPlaceholder(value: Tensor<TFloat32>? = null): Output<TFloat32> {
        val ph = Placeholder.create(scope, TFloat32.DTYPE)
        val out = ph.asOutput() as Output<TFloat32>
        inputs[out.op().name()] = value
        return out
    }

    fun Tensor<TFloat32>.constant(): Output<TFloat32> {
        return scope.env().opBuilder("Const", "MyConst")
                .setAttr("dtype", this.dataType())
                .setAttr("value", this).build().output<TFloat32>(0)
    }

    @JvmName("constantInt")
    fun Tensor<TInt32>.constant(): Output<TInt32> {
        return scope.env().opBuilder("Const", "MyConst")
                .setAttr("dtype", this.dataType())
                .setAttr("value", this).build().output<TInt32>(0)
    }

    fun arrayConstant(value: LongArray): Output<TInt64> {
        val cl = Constant.vectorOf(scope, value)
        return cl.asOutput()
    }

    fun arrayConstant(value: IntArray): Output<TInt32> {
        val cl = Constant.vectorOf(scope, value)
        return cl.asOutput()
    }

    fun arrayConstant(value: Array<LongArray>): Output<TInt64> {
        val cl = Constant.tensorOf(scope, value)
        return cl.asOutput()
    }

    fun arrayConstant(value: Array<FloatArray>): Output<TFloat32> {
        val cl = Constant.tensorOf(scope, value)
        return cl.asOutput()
    }

    fun scalarConstant(value: Int): Output<TInt32> {
        val cl = Constant.scalarOf(scope, value)
        return cl.asOutput()
    }

    fun scalarConstant(value: Long): Output<TInt64> {
        val cl = Constant.scalarOf(scope, value)
        return cl.asOutput()
    }

    fun scalarConstant(value: Float): Output<TFloat32> {
        val cl = Constant.scalarOf(scope, value)
        return cl.asOutput()
    }

    fun scalarConstant(value: Double): Output<TFloat64> {
        val cl = Constant.scalarOf(scope, value)
        return cl.asOutput()
    }

    fun Int.constant(): Output<TInt32> = scalarConstant(this)
    fun Long.constant(): Output<TInt64> = scalarConstant(this)
    fun LongArray.constant(): Output<TInt64> = arrayConstant(this)
    fun IntArray.constant(): Output<TInt32> = arrayConstant(this)


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

}

fun graph(builder: GraphBuilder.() -> Unit): Graph {
    val graph = Graph()
    val gb = GraphBuilder(graph)
    gb.builder()
    return graph
}