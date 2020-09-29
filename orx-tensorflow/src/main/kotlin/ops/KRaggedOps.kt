package org.openrndr.extra.tensorflow.ops

import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.RaggedOps
import org.tensorflow.op.ragged.RaggedBincount
import org.tensorflow.types.TInt64
import org.tensorflow.types.family.TNumber

interface KRaggedOps {
    val raggedOps: RaggedOps

    fun <U : TNumber, T : TNumber> raggedBincount(splits : Operand<TInt64>, values : Operand<T>, size : Operand<T>, weights : Operand<U>, options : RaggedBincount.Options) : Output<U> {
        val op = raggedOps.raggedBincount(splits, values, size, weights, options)
        return op.asOutput()
    }
}