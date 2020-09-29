package org.openrndr.extra.tensorflow.ops

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.QuantizationOps
import org.tensorflow.op.quantization.*
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

interface KQuantizationOps {
    val quantizationOps: QuantizationOps

    fun <T : TType> dequantize(input : Operand<T>, minRange : Operand<TFloat32>, maxRange : Operand<TFloat32>, options : Dequantize.Options) : Output<TFloat32> {
        val op = quantizationOps.dequantize(input, minRange, maxRange, options)
        return op.asOutput()
    }

    fun <U : TNumber, T : TType> dequantize(input : Operand<T>, minRange : Operand<TFloat32>, maxRange : Operand<TFloat32>, dtype : DataType<U>, options : Dequantize.Options) : Output<U> {
        val op = quantizationOps.dequantize(input, minRange, maxRange, dtype, options)
        return op.asOutput()
    }

    fun fakeQuantWithMinMaxArgs(inputs : Operand<TFloat32>, options : FakeQuantWithMinMaxArgs.Options) : Output<TFloat32> {
        val op = quantizationOps.fakeQuantWithMinMaxArgs(inputs, options)
        return op.asOutput()
    }

    fun fakeQuantWithMinMaxArgsGradient(gradients : Operand<TFloat32>, inputs : Operand<TFloat32>, options : FakeQuantWithMinMaxArgsGradient.Options) : Output<TFloat32> {
        val op = quantizationOps.fakeQuantWithMinMaxArgsGradient(gradients, inputs, options)
        return op.asOutput()
    }

    fun fakeQuantWithMinMaxVars(inputs : Operand<TFloat32>, min : Operand<TFloat32>, max : Operand<TFloat32>, options : FakeQuantWithMinMaxVars.Options) : Output<TFloat32> {
        val op = quantizationOps.fakeQuantWithMinMaxVars(inputs, min, max, options)
        return op.asOutput()
    }

    fun fakeQuantWithMinMaxVarsPerChannel(inputs : Operand<TFloat32>, min : Operand<TFloat32>, max : Operand<TFloat32>, options : FakeQuantWithMinMaxVarsPerChannel.Options) : Output<TFloat32> {
        val op = quantizationOps.fakeQuantWithMinMaxVarsPerChannel(inputs, min, max, options)
        return op.asOutput()
    }

    fun <T : TNumber> quantizeAndDequantize(input : Operand<T>, inputMin : Operand<T>, inputMax : Operand<T>, numBits : Operand<TInt32>, options : QuantizeAndDequantize.Options) : Output<T> {
        val op = quantizationOps.quantizeAndDequantize(input, inputMin, inputMax, numBits, options)
        return op.asOutput()
    }
}