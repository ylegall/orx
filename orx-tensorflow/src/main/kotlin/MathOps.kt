import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.MathOps
import org.tensorflow.op.Scope
import org.tensorflow.op.math.*
import org.tensorflow.types.*
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

interface Scoped {
    val scope: Scope
}


interface KMathOps : Scoped {

    val MathOps: MathOps

    fun <T : TNumber> abs(x: Operand<T>): Output<T> {
        val op = MathOps.abs(x)
        return op.asOutput()
    }

//    fun <T : TType> accumulateN(inputs: Iterable<Operand<T>>, shape: Shape): Output<T> {
//        val op = MathOps.accumulateN(inputs, shape)
//        return op.asOutput()
//    }

    fun <T : TType> acos(x: Operand<T>): Output<T> {
        val op = MathOps.acos(x)
        return op.asOutput()
    }

    fun <T : TType> acosh(x: Operand<T>): Output<T> {
        val op = MathOps.acosh(x)
        return op.asOutput()
    }

    fun <T : TType> add(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.add(x, y)
        return op.asOutput()
    }

    fun <T : TType> addN(inputs: Iterable<Operand<T>>): Output<T> {
        val op = MathOps.addN(inputs)
        return op.asOutput()
    }

    fun <T : TType> angle(input: Operand<T>): Output<U> {
        val op = MathOps.angle(input)
        return op.asOutput()
    }

    fun <U : TNumber, T : TType> angle(input: Operand<T>, Tout: DataType<U>): Output<U> {
        val op = MathOps.angle(input, Tout)
        return op.asOutput()
    }

    fun <T : TType> approximateEqual(x: Operand<T>, y: Operand<T>, options: ApproximateEqual.Options): Output<TBool> {
        val op = MathOps.approximateEqual(x, y, options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> argMax(input: Operand<T>, dimension: Operand<U>): Output<T> {
        val op = MathOps.argMax(input, dimension)
        return op.asOutput()
    }

    fun <V : TNumber, T : TType, U : TNumber> argMax(input: Operand<T>, dimension: Operand<U>, outputType: DataType<V>): Output<V> {
        val op = MathOps.argMax(input, dimension, outputType)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> argMin(input: Operand<T>, dimension: Operand<U>): Output<T> {
        val op = MathOps.argMin(input, dimension)
        return op.asOutput()
    }

    fun <V : TNumber, T : TType, U : TNumber> argMin(input: Operand<T>, dimension: Operand<U>, outputType: DataType<V>): Output<V> {
        val op = MathOps.argMin(input, dimension, outputType)
        return op.asOutput()
    }

    fun <T : TType> asin(x: Operand<T>): Output<T> {
        val op = MathOps.asin(x)
        return op.asOutput()
    }

    fun <T : TType> asinh(x: Operand<T>): Output<T> {
        val op = MathOps.asinh(x)
        return op.asOutput()
    }

    fun <T : TType> atan(x: Operand<T>): Output<T> {
        val op = MathOps.atan(x)
        return op.asOutput()
    }

    fun <T : TNumber> atan2(y: Operand<T>, x: Operand<T>): Output<T> {
        val op = MathOps.atan2(y, x)
        return op.asOutput()
    }

    fun <T : TType> atanh(x: Operand<T>): Output<T> {
        val op = MathOps.atanh(x)
        return op.asOutput()
    }

    fun <T : TNumber> betainc(a: Operand<T>, b: Operand<T>, x: Operand<T>): Output<T> {
        val op = MathOps.betainc(a, b, x)
        return op.asOutput()
    }

    fun <T : TNumber> bincount(arr: Operand<TInt32>, size: Operand<TInt32>, weights: Operand<T>): Output<T> {
        val op = MathOps.bincount(arr, size, weights)
        return op.asOutput()
    }

    fun <T : TNumber> ceil(x: Operand<T>): Output<T> {
        val op = MathOps.ceil(x)
        return op.asOutput()
    }

    fun <T : TType> compareAndBitpack(input: Operand<T>, threshold: Operand<T>): Output<TUint8> {
        val op = MathOps.compareAndBitpack(input, threshold)
        return op.asOutput()
    }

    fun <T : TType> complexAbs(x: Operand<T>): Output<U> {
        val op = MathOps.complexAbs(x)
        return op.asOutput()
    }

    fun <U : TNumber, T : TType> complexAbs(x: Operand<T>, Tout: DataType<U>): Output<U> {
        val op = MathOps.complexAbs(x, Tout)
        return op.asOutput()
    }

    fun <T : TType> conj(input: Operand<T>): Output<T> {
        val op = MathOps.conj(input)
        return op.asOutput()
    }

    fun <T : TType> cos(x: Operand<T>): Output<T> {
        val op = MathOps.cos(x)
        return op.asOutput()
    }

    fun <T : TType> cosh(x: Operand<T>): Output<T> {
        val op = MathOps.cosh(x)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> cumprod(x: Operand<T>, axis: Operand<U>, options: Cumprod.Options): Output<T> {
        val op = MathOps.cumprod(x, axis, options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> cumsum(x: Operand<T>, axis: Operand<U>, options: Cumsum.Options): Output<T> {
        val op = MathOps.cumsum(x, axis, options)
        return op.asOutput()
    }

    fun <U : TNumber, T : TNumber> denseBincount(input: Operand<T>, size: Operand<T>, weights: Operand<U>, options: DenseBincount.Options): Output<U> {
        val op = MathOps.denseBincount(input, size, weights, options)
        return op.asOutput()
    }

    fun <T : TNumber> digamma(x: Operand<T>): Output<T> {
        val op = MathOps.digamma(x)
        return op.asOutput()
    }

    fun <T : TType> div(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.div(x, y)
        return op.asOutput()
    }

    fun <T : TType> divNoNan(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.divNoNan(x, y)
        return op.asOutput()
    }

    fun <T : TType> equal(x: Operand<T>, y: Operand<T>, options: Equal.Options): Output<TBool> {
        val op = MathOps.equal(x, y, options)
        return op.asOutput()
    }

    fun <T : TNumber> erf(x: Operand<T>): Output<T> {
        val op = MathOps.erf(x)
        return op.asOutput()
    }

    fun <T : TNumber> erfc(x: Operand<T>): Output<T> {
        val op = MathOps.erfc(x)
        return op.asOutput()
    }

    fun <T : TNumber> erfinv(x: Operand<T>): Output<T> {
        val op = MathOps.erfinv(x)
        return op.asOutput()
    }

    fun <T : TType> exp(x: Operand<T>): Output<T> {
        val op = MathOps.exp(x)
        return op.asOutput()
    }

    fun <T : TType> expm1(x: Operand<T>): Output<T> {
        val op = MathOps.expm1(x)
        return op.asOutput()
    }

    fun  fact(): Output<TString> {
        val op = MathOps.fact()
        return op.asOutput()
    }

    fun <T : TNumber> floor(x: Operand<T>): Output<T> {
        val op = MathOps.floor(x)
        return op.asOutput()
    }

    fun <T : TType> floorDiv(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.floorDiv(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> floorMod(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.floorMod(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> greater(x: Operand<T>, y: Operand<T>): Output<TBool> {
        val op = MathOps.greater(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> greaterEqual(x: Operand<T>, y: Operand<T>): Output<TBool> {
        val op = MathOps.greaterEqual(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> igamma(a: Operand<T>, x: Operand<T>): Output<T> {
        val op = MathOps.igamma(a, x)
        return op.asOutput()
    }

    fun <T : TNumber> igammac(a: Operand<T>, x: Operand<T>): Output<T> {
        val op = MathOps.igammac(a, x)
        return op.asOutput()
    }

    fun <T : TType> imag(input: Operand<T>): Output<TFloat32> {
        val op = MathOps.imag(input)
        return op.asOutput()
    }

    fun <U : TNumber, T : TType> imag(input: Operand<T>, Tout: DataType<U>): Output<U> {
        val op = MathOps.imag(input, Tout)
        return op.asOutput()
    }

    fun <T : TNumber> invertPermutation(x: Operand<T>): Output<T> {
        val op = MathOps.invertPermutation(x)
        return op.asOutput()
    }

    fun <T : TNumber> isFinite(x: Operand<T>): Output<TBool> {
        val op = MathOps.isFinite(x)
        return op.asOutput()
    }

    fun <T : TNumber> isInf(x: Operand<T>): Output<TBool> {
        val op = MathOps.isInf(x)
        return op.asOutput()
    }

    fun <T : TNumber> isNan(x: Operand<T>): Output<TBool> {
        val op = MathOps.isNan(x)
        return op.asOutput()
    }

    fun <T : TNumber> less(x: Operand<T>, y: Operand<T>): Output<TBool> {
        val op = MathOps.less(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> lessEqual(x: Operand<T>, y: Operand<T>): Output<TBool> {
        val op = MathOps.lessEqual(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> lgamma(x: Operand<T>): Output<T> {
        val op = MathOps.lgamma(x)
        return op.asOutput()
    }

    fun <T : TType> log(x: Operand<T>): Output<T> {
        val op = MathOps.log(x)
        return op.asOutput()
    }

    fun <T : TType> log1p(x: Operand<T>): Output<T> {
        val op = MathOps.log1p(x)
        return op.asOutput()
    }

    fun <> logicalAnd(x: Operand<TBool>, y: Operand<TBool>): Output<TBool> {
        val op = MathOps.logicalAnd(x, y)
        return op.asOutput()
    }

    fun <> logicalNot(x: Operand<TBool>): Output<TBool> {
        val op = MathOps.logicalNot(x)
        return op.asOutput()
    }

    fun <> logicalOr(x: Operand<TBool>, y: Operand<TBool>): Output<TBool> {
        val op = MathOps.logicalOr(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> maximum(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.maximum(x, y)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> mean(input: Operand<T>, axis: Operand<U>, options: Mean.Options): Output<T> {
        val op = MathOps.mean(input, axis, options)
        return op.asOutput()
    }

    fun <T : TNumber> minimum(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.minimum(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> mod(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.mod(x, y)
        return op.asOutput()
    }

    fun <T : TType> mul(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.mul(x, y)
        return op.asOutput()
    }

    fun <T : TType> mulNoNan(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.mulNoNan(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> ndtri(x: Operand<T>): Output<T> {
        val op = MathOps.ndtri(x)
        return op.asOutput()
    }

    fun <T : TType> neg(x: Operand<T>): Output<T> {
        val op = MathOps.neg(x)
        return op.asOutput()
    }

    fun <T : TNumber> nextAfter(x1: Operand<T>, x2: Operand<T>): Output<T> {
        val op = MathOps.nextAfter(x1, x2)
        return op.asOutput()
    }

    fun <T : TType> notEqual(x: Operand<T>, y: Operand<T>, options: NotEqual.Options): Output<TBool> {
        val op = MathOps.notEqual(x, y, options)
        return op.asOutput()
    }

    fun <T : TNumber> polygamma(a: Operand<T>, x: Operand<T>): Output<T> {
        val op = MathOps.polygamma(a, x)
        return op.asOutput()
    }

    fun <T : TNumber> populationCount(x: Operand<T>): Output<TUint8> {
        val op = MathOps.populationCount(x)
        return op.asOutput()
    }

    fun <T : TType> pow(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.pow(x, y)
        return op.asOutput()
    }

    fun <T : TType> real(input: Operand<T>): Output<TFloat32> {
        val op = MathOps.real(input)
        return op.asOutput()
    }

    fun <U : TNumber, T : TType> real(input: Operand<T>, Tout: DataType<U>): Output<U> {
        val op = MathOps.real(input, Tout)
        return op.asOutput()
    }

    fun <T : TType> realDiv(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.realDiv(x, y)
        return op.asOutput()
    }

    fun <T : TType> reciprocal(x: Operand<T>): Output<T> {
        val op = MathOps.reciprocal(x)
        return op.asOutput()
    }

    fun <T : TNumber> rint(x: Operand<T>): Output<T> {
        val op = MathOps.rint(x)
        return op.asOutput()
    }

    fun <T : TType> round(x: Operand<T>): Output<T> {
        val op = MathOps.round(x)
        return op.asOutput()
    }

    fun <T : TType> rsqrt(x: Operand<T>): Output<T> {
        val op = MathOps.rsqrt(x)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> segmentMax(data: Operand<T>, segmentIds: Operand<U>): Output<T> {
        val op = MathOps.segmentMax(data, segmentIds)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> segmentMean(data: Operand<T>, segmentIds: Operand<U>): Output<T> {
        val op = MathOps.segmentMean(data, segmentIds)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> segmentMin(data: Operand<T>, segmentIds: Operand<U>): Output<T> {
        val op = MathOps.segmentMin(data, segmentIds)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> segmentProd(data: Operand<T>, segmentIds: Operand<U>): Output<T> {
        val op = MathOps.segmentProd(data, segmentIds)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> segmentSum(data: Operand<T>, segmentIds: Operand<U>): Output<T> {
        val op = MathOps.segmentSum(data, segmentIds)
        return op.asOutput()
    }

    fun <T : TType> sigmoid(x: Operand<T>): Output<T> {
        val op = MathOps.sigmoid(x)
        return op.asOutput()
    }

    fun <T : TType> sign(x: Operand<T>): Output<T> {
        val op = MathOps.sign(x)
        return op.asOutput()
    }

    fun <T : TType> sin(x: Operand<T>): Output<T> {
        val op = MathOps.sin(x)
        return op.asOutput()
    }

    fun <T : TType> sinh(x: Operand<T>): Output<T> {
        val op = MathOps.sinh(x)
        return op.asOutput()
    }

    fun <T : TNumber> softplus(features: Operand<T>): Output<T> {
        val op = MathOps.softplus(features)
        return op.asOutput()
    }

    fun <T : TType> sqrt(x: Operand<T>): Output<T> {
        val op = MathOps.sqrt(x)
        return op.asOutput()
    }

    fun <T : TType> square(x: Operand<T>): Output<T> {
        val op = MathOps.square(x)
        return op.asOutput()
    }

    fun <T : TType> squaredDifference(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.squaredDifference(x, y)
        return op.asOutput()
    }

    fun <T : TType> sub(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.sub(x, y)
        return op.asOutput()
    }

    fun <T : TType> tan(x: Operand<T>): Output<T> {
        val op = MathOps.tan(x)
        return op.asOutput()
    }

    fun <T : TType> tanh(x: Operand<T>): Output<T> {
        val op = MathOps.tanh(x)
        return op.asOutput()
    }

    fun <T : TType> truncateDiv(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.truncateDiv(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> truncateMod(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.truncateMod(x, y)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber, V : TNumber> unsortedSegmentMax(data: Operand<T>, segmentIds: Operand<U>, numSegments: Operand<V>): Output<T> {
        val op = MathOps.unsortedSegmentMax(data, segmentIds, numSegments)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber, V : TNumber> unsortedSegmentMin(data: Operand<T>, segmentIds: Operand<U>, numSegments: Operand<V>): Output<T> {
        val op = MathOps.unsortedSegmentMin(data, segmentIds, numSegments)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber, V : TNumber> unsortedSegmentProd(data: Operand<T>, segmentIds: Operand<U>, numSegments: Operand<V>): Output<T> {
        val op = MathOps.unsortedSegmentProd(data, segmentIds, numSegments)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber, V : TNumber> unsortedSegmentSum(data: Operand<T>, segmentIds: Operand<U>, numSegments: Operand<V>): Output<T> {
        val op = MathOps.unsortedSegmentSum(data, segmentIds, numSegments)
        return op.asOutput()
    }

    fun <T : TType> xdivy(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.xdivy(x, y)
        return op.asOutput()
    }

    fun <T : TType> xlog1py(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.xlog1py(x, y)
        return op.asOutput()
    }

    fun <T : TType> xlogy(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = MathOps.xlogy(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> zeta(x: Operand<T>, q: Operand<T>): Output<T> {
        val op = MathOps.zeta(x, q)
        return op.asOutput()
    }
}
