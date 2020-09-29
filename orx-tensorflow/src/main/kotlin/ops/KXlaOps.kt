package org.openrndr.extra.tensorflow.ops

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.XlaOps
import org.tensorflow.types.TBfloat16
import org.tensorflow.types.TInt32
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

interface KXlaOps {
    val xlaOps: XlaOps

    fun <T : TType> clusterOutput(input : Operand<T>) : Output<T> {
        val op = xlaOps.clusterOutput(input)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> conv(lhs : Operand<T>, rhs : Operand<T>, windowStrides : Operand<U>, padding : Operand<U>, lhsDilation : Operand<U>, rhsDilation : Operand<U>, featureGroupCount : Operand<U>, dimensionNumbers : String, precisionConfig : String) : Output<T> {
        val op = xlaOps.conv(lhs, rhs, windowStrides, padding, lhsDilation, rhsDilation, featureGroupCount, dimensionNumbers, precisionConfig)
        return op.asOutput()
    }

    fun dequantize(input : Operand<*>, minRange : Float, maxRange : Float, mode : String, transposeOutput : Boolean) : Output<TBfloat16> {
        val op = xlaOps.dequantize(input, minRange, maxRange, mode, transposeOutput)
        return op.asOutput()
    }

    fun <T : TType> dot(lhs : Operand<T>, rhs : Operand<T>, dimensionNumbers : String, precisionConfig : String) : Output<T> {
        val op = xlaOps.dot(lhs, rhs, dimensionNumbers, precisionConfig)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> dynamicSlice(input : Operand<T>, startIndices : Operand<U>, sizeIndices : Operand<U>) : Output<T> {
        val op = xlaOps.dynamicSlice(input, startIndices, sizeIndices)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> dynamicUpdateSlice(input : Operand<T>, update : Operand<T>, indices : Operand<U>) : Output<T> {
        val op = xlaOps.dynamicUpdateSlice(input, update, indices)
        return op.asOutput()
    }

    fun <T : TType> einsum(a : Operand<T>, b : Operand<T>, equation : String) : Output<T> {
        val op = xlaOps.einsum(a, b, equation)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> gather(operand : Operand<T>, startIndices : Operand<U>, sliceSizes : Operand<U>, dimensionNumbers : String, indicesAreSorted : Boolean) : Output<T> {
        val op = xlaOps.gather(operand, startIndices, sliceSizes, dimensionNumbers, indicesAreSorted)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> pad(input : Operand<T>, paddingValue : Operand<T>, paddingLow : Operand<U>, paddingHigh : Operand<U>, paddingInterior : Operand<U>) : Output<T> {
        val op = xlaOps.pad(input, paddingValue, paddingLow, paddingHigh, paddingInterior)
        return op.asOutput()
    }

    fun <T : TType> recv(dtype : DataType<T>, tensorName : String, shape : Shape) : Output<T> {
        val op = xlaOps.recv(dtype, tensorName, shape)
        return op.asOutput()
    }

    fun replicaId() : Output<TInt32> {
        val op = xlaOps.replicaId()
        return op.asOutput()
    }

    fun <T : TType> sharding(input : Operand<T>) : Output<T> {
        val op = xlaOps.sharding(input)
        return op.asOutput()
    }

    fun <T : TType> sort(input : Operand<T>) : Output<T> {
        val op = xlaOps.sort(input)
        return op.asOutput()
    }
}