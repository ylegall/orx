package org.openrndr.extra.tensorflow.ops

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.DtypesOps
import org.tensorflow.op.dtypes.AsString
import org.tensorflow.op.dtypes.Cast
import org.tensorflow.types.TString
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

interface KDtypesOps {
    val dtypesOps: DtypesOps

    fun <T : TType> asString(input : Operand<T>, options : AsString.Options) : Output<TString> {
        val op = dtypesOps.asString(input, options)
        return op.asOutput()
    }

    fun <U : TType, T : TType> cast(x : Operand<T>, DstT : DataType<U>, options : Cast.Options) : Output<U> {
        val op = dtypesOps.cast(x, DstT, options)
        return op.asOutput()
    }

    fun <U : TType, T : TNumber> complex(real : Operand<T>, imag : Operand<T>, Tout : DataType<U>) : Output<U> {
        val op = dtypesOps.complex(real, imag, Tout)
        return op.asOutput()
    }
}