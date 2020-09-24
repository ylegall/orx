import org.tensorflow.DataType
import org.tensorflow.Output
import org.tensorflow.Tensor
import org.tensorflow.op.Scope
import org.tensorflow.op.core.*
import org.tensorflow.op.dtypes.Cast
import org.tensorflow.op.image.CropAndResize
import org.tensorflow.op.image.ResizeBilinear
import org.tensorflow.op.linalg.MatMul
import org.tensorflow.op.math.*
import org.tensorflow.types.*
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

class GraphBuilder(val scope: Scope) {
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

    inline operator fun <reified T : TType> Output<T>.plus(add: Output<T>): Output<T> {
        return Add.create(scope, this, add).op().output(0)
    }

    fun range(start: Int, limit: Int, delta: Int): Output<TInt32> {
        val startConstant = (start.constant()).op().output<TInt32>(0)
        val limitConstant = (limit.constant()).op().output<TInt32>(0)
        val deltaConstant = (delta.constant()).op().output<TInt32>(0)
        return Range.create(scope, startConstant, limitConstant, deltaConstant).op().output(0)
    }

    fun range(start: Long, limit: Long, delta: Long): Output<TInt64> {
        val startConstant = (start.constant()).op().output<TInt64>(0)
        val limitConstant = (limit.constant()).op().output<TInt64>(0)
        val deltaConstant = (delta.constant()).op().output<TInt64>(0)
        return Range.create(scope, startConstant, limitConstant, deltaConstant).op().output(0)
    }

    fun <T : TType> Output<T>.argMax(depth: Long = 0): Output<T> {
        val depthConstant = depth.constant()
        return ArgMax.create(scope, this, depthConstant).op().output(0)
    }

    fun <T : TType> Output<T>.expandDims(axis: Long = 0): Output<T> {
        val depthConstant = axis.constant()
        return ExpandDims.create(scope, this, depthConstant).asOutput()
    }

    fun <T : TType> Output<T>.reverse(axis: Long): Output<T> {
        return Reverse.create(scope, this, longArrayOf(axis).constant()).asOutput()
    }

    fun <T : TType> Output<T>.sigmoid(): Output<T> {
        return Sigmoid.create(scope, this).asOutput()
    }

    fun Output<TFloat32>.resizeBilinear(size: IntArray): Output<TFloat32> {
        return ResizeBilinear.create(scope, this, size.constant()).asOutput()
    }

    fun <T : TType> Output<T>.slice(start: LongArray, size: LongArray): Output<T> {
        return Slice.create(scope, this, start.constant(), size.constant()).asOutput()
    }

    fun <T : TType> Output<T>.castToFloat32(): Output<TFloat32> {
        return Cast.create(scope, this, TFloat32.DTYPE).asOutput() as Output<TFloat32>
    }

    fun <T : TType> Output<T>.castToInt32(): Output<TInt32> {
        return Cast.create(scope, this, TInt32.DTYPE).asOutput() as Output<TInt32>
    }

    fun Output<TFloat32>.greater(comp: Float): Output<TBool> {
        return Greater.create(scope,this, scalarConstant(comp)).asOutput()
    }

    fun <T: TNumber> Output<T>.oneHot(size:Int, onValue:Float = 1.0f, offValue:Float = 0.0f) : Output<TFloat32> {
        return OneHot.create(scope, this, scalarConstant(size), scalarConstant(onValue), scalarConstant(offValue)).asOutput()
    }

    fun <T : TType> Output<T>.squeeze(axis: LongArray): Output<T> {
        val options = Squeeze.axis(axis.toList())
        return Squeeze.create(scope, this, options).asOutput()
    }
    fun <T : TNumber> Output<T>.cropAndResize(
            boxes: Array<FloatArray>,
            boxIndices: IntArray,
            size: IntArray
    ): Output<TFloat32> {
        return CropAndResize.create(scope, this, arrayConstant(boxes), boxIndices.constant(), size.constant()).asOutput()
    }
}

fun graph(scope: Scope, builder: GraphBuilder.() -> Unit) {
    val gb = GraphBuilder(scope)
    gb.builder()
    //return ExecutableGraph(gb.scope, gb.inputs, gb.outputs)
}