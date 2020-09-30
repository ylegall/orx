package org.openrndr.extra.tensorflow.ops

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.RandomOps
import org.tensorflow.op.random.*
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32
import org.tensorflow.types.TInt64
import org.tensorflow.types.TString
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

interface KRandomOps {
    val randomOps: RandomOps

    fun <T : TNumber> multinomial(logits: Operand<T>, numSamples: Operand<TInt32>, vararg options: Multinomial.Options): Output<TInt64> {
        val op = randomOps.multinomial(logits, numSamples, *options)
        return op.asOutput()
    }

    fun <U : TNumber, T : TNumber> multinomial(logits: Operand<T>, numSamples: Operand<TInt32>, outputDtype: DataType<U>, vararg options: Multinomial.Options): Output<U> {
        val op = randomOps.multinomial(logits, numSamples, outputDtype, *options)
        return op.asOutput()
    }

    fun <U : TNumber, T : TNumber> parameterizedTruncatedNormal(shape: Operand<T>, means: Operand<U>, stdevs: Operand<U>, minvals: Operand<U>, maxvals: Operand<U>, vararg options: ParameterizedTruncatedNormal.Options): Output<U> {
        val op = randomOps.parameterizedTruncatedNormal(shape, means, stdevs, minvals, maxvals, *options)
        return op.asOutput()
    }

    fun <U : TNumber, T : TNumber> randomGamma(shape: Operand<T>, alpha: Operand<U>, vararg options: RandomGamma.Options): Output<U> {
        val op = randomOps.randomGamma(shape, alpha, *options)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> randomPoisson(shape: Operand<T>, rate: Operand<U>, vararg options: RandomPoisson.Options): Output<TInt64> {
        val op = randomOps.randomPoisson(shape, rate, *options)
        return op.asOutput()
    }

    fun <V : TNumber, T : TNumber, U : TNumber> randomPoisson(shape: Operand<T>, rate: Operand<U>, dtype: DataType<V>, vararg options: RandomPoisson.Options): Output<V> {
        val op = randomOps.randomPoisson(shape, rate, dtype, *options)
        return op.asOutput()
    }

    fun <T : TType> randomShuffle(value: Operand<T>, vararg options: RandomShuffle.Options): Output<T> {
        val op = randomOps.randomShuffle(value, *options)
        return op.asOutput()
    }

    fun <U : TNumber, T : TNumber> randomStandardNormal(shape: Operand<T>, dtype: DataType<U>, vararg options: RandomStandardNormal.Options): Output<U> {
        val op = randomOps.randomStandardNormal(shape, dtype, *options)
        return op.asOutput()
    }

    fun <U : TNumber, T : TNumber> randomUniform(shape: Operand<T>, dtype: DataType<U>, vararg options: RandomUniform.Options): Output<U> {
        val op = randomOps.randomUniform(shape, dtype, *options)
        return op.asOutput()
    }

    fun <U : TNumber, T : TNumber> randomUniformInt(shape: Operand<T>, minval: Operand<U>, maxval: Operand<U>, vararg options: RandomUniformInt.Options): Output<U> {
        val op = randomOps.randomUniformInt(shape, minval, maxval, *options)
        return op.asOutput()
    }

    fun recordInput(filePattern: String, vararg options: RecordInput.Options): Output<TString> {
        val op = randomOps.recordInput(filePattern, *options)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> statefulRandomBinomial(resource: Operand<*>, algorithm: Operand<TInt64>, shape: Operand<T>, counts: Operand<U>, probs: Operand<U>): Output<TInt64> {
        val op = randomOps.statefulRandomBinomial(resource, algorithm, shape, counts, probs)
        return op.asOutput()
    }

    fun <V : TNumber, T : TNumber, U : TNumber> statefulRandomBinomial(resource: Operand<*>, algorithm: Operand<TInt64>, shape: Operand<T>, counts: Operand<U>, probs: Operand<U>, dtype: DataType<V>): Output<V> {
        val op = randomOps.statefulRandomBinomial(resource, algorithm, shape, counts, probs, dtype)
        return op.asOutput()
    }

    fun <T : TType> statefulStandardNormal(resource: Operand<*>, algorithm: Operand<TInt64>, shape: Operand<T>): Output<TFloat32> {
        val op = randomOps.statefulStandardNormal(resource, algorithm, shape)
        return op.asOutput()
    }

    fun <U : TType, T : TType> statefulStandardNormal(resource: Operand<*>, algorithm: Operand<TInt64>, shape: Operand<T>, dtype: DataType<U>): Output<U> {
        val op = randomOps.statefulStandardNormal(resource, algorithm, shape, dtype)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> statelessMultinomial(logits: Operand<T>, numSamples: Operand<TInt32>, seed: Operand<U>): Output<TInt64> {
        val op = randomOps.statelessMultinomial(logits, numSamples, seed)
        return op.asOutput()
    }

    fun <V : TNumber, T : TNumber, U : TNumber> statelessMultinomial(logits: Operand<T>, numSamples: Operand<TInt32>, seed: Operand<U>, outputDtype: DataType<V>): Output<V> {
        val op = randomOps.statelessMultinomial(logits, numSamples, seed, outputDtype)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> statelessRandomNormal(shape: Operand<T>, seed: Operand<U>): Output<TFloat32> {
        val op = randomOps.statelessRandomNormal(shape, seed)
        return op.asOutput()
    }

    fun <V : TNumber, T : TNumber, U : TNumber> statelessRandomNormal(shape: Operand<T>, seed: Operand<U>, dtype: DataType<V>): Output<V> {
        val op = randomOps.statelessRandomNormal(shape, seed, dtype)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> statelessRandomUniform(shape: Operand<T>, seed: Operand<U>): Output<TFloat32> {
        val op = randomOps.statelessRandomUniform(shape, seed)
        return op.asOutput()
    }

    fun <V : TNumber, T : TNumber, U : TNumber> statelessRandomUniform(shape: Operand<T>, seed: Operand<U>, dtype: DataType<V>): Output<V> {
        val op = randomOps.statelessRandomUniform(shape, seed, dtype)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> statelessTruncatedNormal(shape: Operand<T>, seed: Operand<U>): Output<TFloat32> {
        val op = randomOps.statelessTruncatedNormal(shape, seed)
        return op.asOutput()
    }

    fun <V : TNumber, T : TNumber, U : TNumber> statelessTruncatedNormal(shape: Operand<T>, seed: Operand<U>, dtype: DataType<V>): Output<V> {
        val op = randomOps.statelessTruncatedNormal(shape, seed, dtype)
        return op.asOutput()
    }

    fun <U : TNumber, T : TNumber> truncatedNormal(shape: Operand<T>, dtype: DataType<U>, vararg options: TruncatedNormal.Options): Output<U> {
        val op = randomOps.truncatedNormal(shape, dtype, *options)
        return op.asOutput()
    }
}