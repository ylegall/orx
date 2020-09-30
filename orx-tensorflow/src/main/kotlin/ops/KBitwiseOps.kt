package org.openrndr.extra.tensorflow.ops

import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.BitwiseOps
import org.tensorflow.types.family.TNumber

interface KBitwiseOps {
    val bitwiseOps: BitwiseOps

    fun <T : TNumber> bitwiseAnd(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = bitwiseOps.bitwiseAnd(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> bitwiseOr(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = bitwiseOps.bitwiseOr(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> bitwiseXor(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = bitwiseOps.bitwiseXor(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> invert(x: Operand<T>): Output<T> {
        val op = bitwiseOps.invert(x)
        return op.asOutput()
    }

    fun <T : TNumber> leftShift(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = bitwiseOps.leftShift(x, y)
        return op.asOutput()
    }

    fun <T : TNumber> rightShift(x: Operand<T>, y: Operand<T>): Output<T> {
        val op = bitwiseOps.rightShift(x, y)
        return op.asOutput()
    }
}
