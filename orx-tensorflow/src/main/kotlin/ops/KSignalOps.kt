package org.openrndr.extra.tensorflow.ops

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.SignalOps
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

interface KSignalOps {
    val signalOps: SignalOps

    fun batchFft(input : Operand<*>) : Output<TType> {
        val op = signalOps.batchFft(input)
        return op.asOutput()
    }

    fun batchFft2d(input : Operand<*>) : Output<TType> {
        val op = signalOps.batchFft2d(input)
        return op.asOutput()
    }

    fun batchFft3d(input : Operand<*>) : Output<TType> {
        val op = signalOps.batchFft3d(input)
        return op.asOutput()
    }

    fun batchIfft(input : Operand<*>) : Output<TType> {
        val op = signalOps.batchIfft(input)
        return op.asOutput()
    }

    fun batchIfft2d(input : Operand<*>) : Output<TType> {
        val op = signalOps.batchIfft2d(input)
        return op.asOutput()
    }

    fun batchIfft3d(input : Operand<*>) : Output<TType> {
        val op = signalOps.batchIfft3d(input)
        return op.asOutput()
    }

    fun <T : TType> fft(input : Operand<T>) : Output<T> {
        val op = signalOps.fft(input)
        return op.asOutput()
    }

    fun <T : TType> fft2d(input : Operand<T>) : Output<T> {
        val op = signalOps.fft2d(input)
        return op.asOutput()
    }

    fun <T : TType> fft3d(input : Operand<T>) : Output<T> {
        val op = signalOps.fft3d(input)
        return op.asOutput()
    }

    fun <T : TType> ifft(input : Operand<T>) : Output<T> {
        val op = signalOps.ifft(input)
        return op.asOutput()
    }

    fun <T : TType> ifft2d(input : Operand<T>) : Output<T> {
        val op = signalOps.ifft2d(input)
        return op.asOutput()
    }

    fun <T : TType> ifft3d(input : Operand<T>) : Output<T> {
        val op = signalOps.ifft3d(input)
        return op.asOutput()
    }

    fun <T : TType> irfft(input : Operand<T>, fftLength : Operand<TInt32>) : Output<TFloat32> {
        val op = signalOps.irfft(input, fftLength)
        return op.asOutput()
    }

    fun <U : TNumber, T : TType> irfft(input : Operand<T>, fftLength : Operand<TInt32>, Treal : DataType<U>) : Output<U> {
        val op = signalOps.irfft(input, fftLength, Treal)
        return op.asOutput()
    }

    fun <T : TType> irfft2d(input : Operand<T>, fftLength : Operand<TInt32>) : Output<TFloat32> {
        val op = signalOps.irfft2d(input, fftLength)
        return op.asOutput()
    }

    fun <U : TNumber, T : TType> irfft2d(input : Operand<T>, fftLength : Operand<TInt32>, Treal : DataType<U>) : Output<U> {
        val op = signalOps.irfft2d(input, fftLength, Treal)
        return op.asOutput()
    }

    fun <T : TType> irfft3d(input : Operand<T>, fftLength : Operand<TInt32>) : Output<TFloat32> {
        val op = signalOps.irfft3d(input, fftLength)
        return op.asOutput()
    }

    fun <U : TNumber, T : TType> irfft3d(input : Operand<T>, fftLength : Operand<TInt32>, Treal : DataType<U>) : Output<U> {
        val op = signalOps.irfft3d(input, fftLength, Treal)
        return op.asOutput()
    }

    fun <U : TType, T : TNumber> rfft(input : Operand<T>, fftLength : Operand<TInt32>, Tcomplex : DataType<U>) : Output<U> {
        val op = signalOps.rfft(input, fftLength, Tcomplex)
        return op.asOutput()
    }

    fun <U : TType, T : TNumber> rfft2d(input : Operand<T>, fftLength : Operand<TInt32>, Tcomplex : DataType<U>) : Output<U> {
        val op = signalOps.rfft2d(input, fftLength, Tcomplex)
        return op.asOutput()
    }

    fun <U : TType, T : TNumber> rfft3d(input : Operand<T>, fftLength : Operand<TInt32>, Tcomplex : DataType<U>) : Output<U> {
        val op = signalOps.rfft3d(input, fftLength, Tcomplex)
        return op.asOutput()
    }
}
