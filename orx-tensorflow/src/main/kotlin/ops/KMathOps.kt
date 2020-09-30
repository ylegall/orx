package org.openrndr.extra.tensorflow.ops

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.MathOps
import org.tensorflow.op.Scope
import org.tensorflow.op.math.*
import org.tensorflow.types.*
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType


interface KMathOps {

    val mathOps: MathOps

    fun <T : TNumber> abs(x: Operand<T>): Output<T> {
        val op = mathOps.abs(x)
        return op.asOutput()
    }

//    fun <T : TType> accumulateN(inputs: Iterable<Operand<T>>, shape: Shape): Output<T> {
//        val op = MathOps.accumulateN(inputs, shape)
//        return op.asOutput()
//    }

    fun <T : TType> acos(x: Operand<T>): Output<T> {
        val op = mathOps.acos(x)
        return op.asOutput()
    }

    fun <T : TType> acosh(x: Operand<T>): Output<T> {
        val op = mathOps.acosh(x)
        return op.asOutput()
    }

    fun <T : TType> add(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.add(x, y)
        return op.asOutput()
    }

    fun <T : TType> addN(inputs: Iterable<Operand<T>>): Output<T> {
        val op = mathOps.addN(inputs)
        return op.asOutput()
    }

    fun <T : TType> angle(input: Operand<T>): Output<TFloat32> {
        val op = mathOps.angle(input)
        return op.asOutput()
    }

    fun <U : TNumber, T : TType> angle(input: Operand<T>, Tout: DataType<U>): Output<U> {
        val op = mathOps.angle(input, Tout)
        return op.asOutput()
    }

    fun <T : TType> approximateEqual(x: Operand<T>, y: Operand<T>, vararg options: ApproximateEqual.Options): Output<TBool> {
        val op = mathOps.approximateEqual(x, y, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> argMax(input: Operand<T>, dimension: Operand<U>): Output<TInt64>? {
        val op = mathOps.argMax(input, dimension)
        return op.asOutput()
    }

    fun <V : TNumber, T : TType, U : TNumber> argMax(input: Operand<T>, dimension: Operand<U>, outputType: DataType<V>): Output<V> {
        val op = mathOps.argMax(input, dimension, outputType)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> argMin(input: Operand<T>, dimension: Operand<U>): Output<TInt64> {
        val op = mathOps.argMin(input, dimension)
        return op.asOutput()
    }

    fun <V : TNumber, T : TType, U : TNumber> argMin(input: Operand<T>, dimension: Operand<U>, outputType: DataType<V>): Output<V> {
        val op = mathOps.argMin(input, dimension, outputType)
        return op.asOutput()
    }

    fun <T : TType> asin(x: Operand<T>): Output<T> {
        val op = mathOps.asin(x)
        return op.asOutput()
    }

    fun <T : TType> asinh(x: Operand<T>): Output<T> {
        val op = mathOps.asinh(x)
        return op.asOutput()
    }

    fun <T : TType> atan(x: Operand<T>): Output<T> {
        val op = mathOps.atan(x)
        return op.asOutput()
    }

    fun <T : TNumber> atan2(y: Operand<T>, x: Operand<T>): Output<T> {
        val op = mathOps.atan2(y, x)
        return op.asOutput()
    }

    fun <T : TType> atanh(x: Operand<T>): Output<T> {
        val op = mathOps.atanh(x)
        return op.asOutput()
    }

    fun <T : TNumber> betainc(a: Operand<T>, b: Operand<T>, x: Operand<T>): Output<T> {
        val op = mathOps.betainc(a, b, x)
        return op.asOutput()
    }

    fun <T : TNumber> bincount(arr: Operand<TInt32>, size: Operand<TInt32>, weights: Operand<T>): Output<T> {
        val op = mathOps.bincount(arr, size, weights)
        return op.asOutput()
    }

    fun <T : TNumber> ceil(x: Operand<T>): Output<T> {
        val op = mathOps.ceil(x)
        return op.asOutput()
    }

    fun <T : TType> compareAndBitpack(input: Operand<T>, threshold: Operand<T>): Output<TUint8> {
        val op = mathOps.compareAndBitpack(input, threshold)
        return op.asOutput()
    }

    fun <T : TType> complexAbs(x: Operand<T>): Output<TFloat32> {
        val op = mathOps.complexAbs(x)
        return op.asOutput()
    }

    fun <U : TNumber, T : TType> complexAbs(x: Operand<T>, Tout: DataType<U>): Output<U> {
        val op = mathOps.complexAbs(x, Tout)
        return op.asOutput()
    }

    fun <T : TType> conj(input: Operand<T>): Output<T> {
        val op = mathOps.conj(input)
        return op.asOutput()
    }

    fun <T : TType> cos(x: Operand<T>): Output<T> {
        val op = mathOps.cos(x)
        return op.asOutput()
    }

    fun <T : TType> cosh(x: Operand<T>): Output<T> {
        val op = mathOps.cosh(x)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> cumprod(x: Operand<T>, axis: Operand<U>, vararg options: Cumprod.Options): Output<T> {
        val op = mathOps.cumprod(x, axis, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> cumsum(x: Operand<T>, axis: Operand<U>, vararg options: Cumsum.Options): Output<T> {
        val op = mathOps.cumsum(x, axis, *options)
        return op.asOutput()
    }

    fun <U : TNumber, T : TNumber> denseBincount(input: Operand<T>, size: Operand<T>, weights: Operand<U>, vararg options: DenseBincount.Options): Output<U> {
        val op = mathOps.denseBincount(input, size, weights, *options)
        return op.asOutput()
    }

    fun <T : TNumber> digamma(x: Operand<T>): Output<T> {
        val op = mathOps.digamma(x)
        return op.asOutput()
    }

    fun <T : TType> div(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.div(x, y)
        return op.asOutput()
    }

    fun <T : TType> divNoNan(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.divNoNan(x, y)
        return op.asOutput()
    }

    fun <T : TType> equal(x: Operand<T>, y: Operand<T>, vararg options: Equal.Options): Output<TBool> {
        val op = mathOps.equal(x, y, *options)
        return op.asOutput()
    }

    fun <T : TNumber> erf(x: Operand<T>): Output<T> {
        val op = mathOps.erf(x)
        return op.asOutput()
    }

    fun <T : TNumber> erfc(x: Operand<T>): Output<T> {
        val op = mathOps.erfc(x)
        return op.asOutput()
    }

    fun <T : TNumber> erfinv(x: Operand<T>): Output<T> {
        val op = mathOps.erfinv(x)
        return op.asOutput()
    }

    fun <T : TType> exp(x: Operand<T>): Output<T> {
        val op = mathOps.exp(x)
        return op.asOutput()
    }

    fun <T : TType> expm1(x: Operand<T>): Output<T> {
        val op = mathOps.expm1(x)
        return op.asOutput()
    }

    fun fact(): Output<TString> {
        val op = mathOps.fact()
        return op.asOutput()
    }

    fun <T : TNumber> floor(x: Operand<T>): Output<T> {
        val op = mathOps.floor(x)
        return op.asOutput()
    }

    fun <T : TType> floorDiv(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.floorDiv(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> floorMod(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.floorMod(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> greater(x: Operand<T>, y: Operand<T>): Output<TBool> {
        val op = mathOps.greater(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> greaterEqual(x: Operand<T>, y: Operand<T>): Output<TBool> {
        val op = mathOps.greaterEqual(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> igamma(a: Operand<T>, x: Operand<T>): Output<T> {
        val op = mathOps.igamma(a, x)
        return op.asOutput()
    }

    fun <T : TNumber> igammac(a: Operand<T>, x: Operand<T>): Output<T> {
        val op = mathOps.igammac(a, x)
        return op.asOutput()
    }

    fun <T : TType> imag(input: Operand<T>): Output<TFloat32> {
        val op = mathOps.imag(input)
        return op.asOutput()
    }

    fun <U : TNumber, T : TType> imag(input: Operand<T>, Tout: DataType<U>): Output<U> {
        val op = mathOps.imag(input, Tout)
        return op.asOutput()
    }

    fun <T : TNumber> invertPermutation(x: Operand<T>): Output<T> {
        val op = mathOps.invertPermutation(x)
        return op.asOutput()
    }

    fun <T : TNumber> isFinite(x: Operand<T>): Output<TBool> {
        val op = mathOps.isFinite(x)
        return op.asOutput()
    }

    fun <T : TNumber> isInf(x: Operand<T>): Output<TBool> {
        val op = mathOps.isInf(x)
        return op.asOutput()
    }

    fun <T : TNumber> isNan(x: Operand<T>): Output<TBool> {
        val op = mathOps.isNan(x)
        return op.asOutput()
    }

    fun <T : TNumber> less(x: Operand<T>, y: Operand<T>): Output<TBool> {
        val op = mathOps.less(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> lessEqual(x: Operand<T>, y: Operand<T>): Output<TBool> {
        val op = mathOps.lessEqual(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> lgamma(x: Operand<T>): Output<T> {
        val op = mathOps.lgamma(x)
        return op.asOutput()
    }

    fun <T : TType> log(x: Operand<T>): Output<T> {
        val op = mathOps.log(x)
        return op.asOutput()
    }

    fun <T : TType> log1p(x: Operand<T>): Output<T> {
        val op = mathOps.log1p(x)
        return op.asOutput()
    }

    fun logicalAnd(x: Operand<TBool>, y: Operand<TBool>): Output<TBool> {
        val op = mathOps.logicalAnd(x, y)
        return op.asOutput()
    }

    fun logicalNot(x: Operand<TBool>): Output<TBool> {
        val op = mathOps.logicalNot(x)
        return op.asOutput()
    }

    fun logicalOr(x: Operand<TBool>, y: Operand<TBool>): Output<TBool> {
        val op = mathOps.logicalOr(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> maximum(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.maximum(x, y)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> mean(input: Operand<T>, axis: Operand<U>, vararg options: Mean.Options): Output<T> {
        val op = mathOps.mean(input, axis, *options)
        return op.asOutput()
    }

    fun <T : TNumber> minimum(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.minimum(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> mod(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.mod(x, y)
        return op.asOutput()
    }

    fun <T : TType> mul(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.mul(x, y)
        return op.asOutput()
    }

    fun <T : TType> mulNoNan(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.mulNoNan(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> ndtri(x: Operand<T>): Output<T> {
        val op = mathOps.ndtri(x)
        return op.asOutput()
    }

    fun <T : TType> neg(x: Operand<T>): Output<T> {
        val op = mathOps.neg(x)
        return op.asOutput()
    }

    fun <T : TNumber> nextAfter(x1: Operand<T>, x2: Operand<T>): Output<T> {
        val op = mathOps.nextAfter(x1, x2)
        return op.asOutput()
    }

    fun <T : TType> notEqual(x: Operand<T>, y: Operand<T>, vararg options: NotEqual.Options): Output<TBool> {
        val op = mathOps.notEqual(x, y, *options)
        return op.asOutput()
    }

    fun <T : TNumber> polygamma(a: Operand<T>, x: Operand<T>): Output<T> {
        val op = mathOps.polygamma(a, x)
        return op.asOutput()
    }

    fun <T : TNumber> populationCount(x: Operand<T>): Output<TUint8> {
        val op = mathOps.populationCount(x)
        return op.asOutput()
    }

    fun <T : TType> pow(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.pow(x, y)
        return op.asOutput()
    }

    fun <T : TType> real(input: Operand<T>): Output<TFloat32> {
        val op = mathOps.real(input)
        return op.asOutput()
    }

    fun <U : TNumber, T : TType> real(input: Operand<T>, Tout: DataType<U>): Output<U> {
        val op = mathOps.real(input, Tout)
        return op.asOutput()
    }

    fun <T : TType> realDiv(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.realDiv(x, y)
        return op.asOutput()
    }

    fun <T : TType> reciprocal(x: Operand<T>): Output<T> {
        val op = mathOps.reciprocal(x)
        return op.asOutput()
    }

    fun <T : TNumber> rint(x: Operand<T>): Output<T> {
        val op = mathOps.rint(x)
        return op.asOutput()
    }

    fun <T : TType> round(x: Operand<T>): Output<T> {
        val op = mathOps.round(x)
        return op.asOutput()
    }

    fun <T : TType> rsqrt(x: Operand<T>): Output<T> {
        val op = mathOps.rsqrt(x)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> segmentMax(data: Operand<T>, segmentIds: Operand<U>): Output<T> {
        val op = mathOps.segmentMax(data, segmentIds)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> segmentMean(data: Operand<T>, segmentIds: Operand<U>): Output<T> {
        val op = mathOps.segmentMean(data, segmentIds)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> segmentMin(data: Operand<T>, segmentIds: Operand<U>): Output<T> {
        val op = mathOps.segmentMin(data, segmentIds)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> segmentProd(data: Operand<T>, segmentIds: Operand<U>): Output<T> {
        val op = mathOps.segmentProd(data, segmentIds)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> segmentSum(data: Operand<T>, segmentIds: Operand<U>): Output<T> {
        val op = mathOps.segmentSum(data, segmentIds)
        return op.asOutput()
    }

    fun <T : TType> sigmoid(x: Operand<T>): Output<T> {
        val op = mathOps.sigmoid(x)
        return op.asOutput()
    }

    fun <T : TType> sign(x: Operand<T>): Output<T> {
        val op = mathOps.sign(x)
        return op.asOutput()
    }

    fun <T : TType> sin(x: Operand<T>): Output<T> {
        val op = mathOps.sin(x)
        return op.asOutput()
    }

    fun <T : TType> sinh(x: Operand<T>): Output<T> {
        val op = mathOps.sinh(x)
        return op.asOutput()
    }

    fun <T : TNumber> softplus(features: Operand<T>): Output<T> {
        val op = mathOps.softplus(features)
        return op.asOutput()
    }

    fun <T : TType> sqrt(x: Operand<T>): Output<T> {
        val op = mathOps.sqrt(x)
        return op.asOutput()
    }

    fun <T : TType> square(x: Operand<T>): Output<T> {
        val op = mathOps.square(x)
        return op.asOutput()
    }

    fun <T : TType> squaredDifference(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.squaredDifference(x, y)
        return op.asOutput()
    }

    fun <T : TType> sub(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.sub(x, y)
        return op.asOutput()
    }

    fun <T : TType> tan(x: Operand<T>): Output<T> {
        val op = mathOps.tan(x)
        return op.asOutput()
    }

    fun <T : TType> tanh(x: Operand<T>): Output<T> {
        val op = mathOps.tanh(x)
        return op.asOutput()
    }

    fun <T : TType> truncateDiv(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.truncateDiv(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> truncateMod(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.truncateMod(x, y)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber, V : TNumber> unsortedSegmentMax(data: Operand<T>, segmentIds: Operand<U>, numSegments: Operand<V>): Output<T> {
        val op = mathOps.unsortedSegmentMax(data, segmentIds, numSegments)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber, V : TNumber> unsortedSegmentMin(data: Operand<T>, segmentIds: Operand<U>, numSegments: Operand<V>): Output<T> {
        val op = mathOps.unsortedSegmentMin(data, segmentIds, numSegments)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber, V : TNumber> unsortedSegmentProd(data: Operand<T>, segmentIds: Operand<U>, numSegments: Operand<V>): Output<T> {
        val op = mathOps.unsortedSegmentProd(data, segmentIds, numSegments)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber, V : TNumber> unsortedSegmentSum(data: Operand<T>, segmentIds: Operand<U>, numSegments: Operand<V>): Output<T> {
        val op = mathOps.unsortedSegmentSum(data, segmentIds, numSegments)
        return op.asOutput()
    }

    fun <T : TType> xdivy(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.xdivy(x, y)
        return op.asOutput()
    }

    fun <T : TType> xlog1py(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.xlog1py(x, y)
        return op.asOutput()
    }

    fun <T : TType> xlogy(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = mathOps.xlogy(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> zeta(x: Operand<T>, q: Operand<T>): Output<T> {
        val op = mathOps.zeta(x, q)
        return op.asOutput()
    }
}
