package org.openrndr.extra.tensorflow.ops

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Output

import org.tensorflow.op.ShapeOps
import org.tensorflow.op.core.Shape
import org.tensorflow.types.TInt32
import org.tensorflow.types.TInt64
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

interface KShapeOps {
    val shapeOps: ShapeOps

    fun append(shape : Shape<TInt64>, lastDimension : Long) : Output<TInt64> {
        val op = shapeOps.append(shape, lastDimension)
        return op.asOutput()
    }

    fun append(shape : Shape<TInt32>, lastDimension : Int) : Output<TInt32>? {
        val op = shapeOps.append(shape, lastDimension)
        return op.asOutput()
    }

    fun <T : TNumber> append(shape : Operand<T>, shapeToAppend : Operand<T>) : Output<T> {
        val op = shapeOps.append(shape, shapeToAppend)
        return op.asOutput()
    }

    fun <T : TType> flatten(operand : Operand<T>) : Output<T> {
        val op = shapeOps.flatten(operand)
        return op.asOutput()
    }

    fun flatten(shape : Shape<TInt32>) : Output<TInt32> {
        val op = shapeOps.flatten(shape)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> flatten(operand : Operand<T>, dType : DataType<U>) : Output<T> {
        val op = shapeOps.flatten(operand, dType)
        return op.asOutput()
    }

    fun <U : TNumber> flatten(shape : Shape<U>, dType : DataType<U>) : Output<U> {
        val op = shapeOps.flatten(shape, dType)
        return op.asOutput()
    }

    fun head(shape : Shape<TInt32>) : Output<TInt32> {
        val op = shapeOps.head(shape)
        return op.asOutput()
    }

    fun <U : TNumber> head(shape : Shape<U>, dType : DataType<U>) : Output<U> {
        val op = shapeOps.head(shape, dType)
        return op.asOutput()
    }

    fun numDimensions(shape : Shape<TInt32>) : Output<TInt32> {
        val op = shapeOps.numDimensions(shape)
        return op.asOutput()
    }

    fun <U : TNumber> numDimensions(shape : Shape<U>, dType : DataType<U>) : Output<U> {
        val op = shapeOps.numDimensions(shape, dType)
        return op.asOutput()
    }

    fun prepend(shape : Shape<TInt64>, firstDimension : Long) : Output<TInt64> {
        val op = shapeOps.prepend(shape, firstDimension)
        return op.asOutput()
    }

    fun prepend(shape : Shape<TInt32>, firstDimension : Int) : Output<TInt32> {
        val op = shapeOps.prepend(shape, firstDimension)
        return op.asOutput()
    }

    fun <T : TNumber> prepend(shape : Operand<T>, shapeToPrepend : Operand<T>) : Output<T> {
        val op = shapeOps.prepend(shape, shapeToPrepend)
        return op.asOutput()
    }

    fun <T : TType> reduceDims(operand : Operand<T>, axis : Operand<TInt32>) : Output<T> {
        val op = shapeOps.reduceDims(operand, axis)
        return op.asOutput()
    }

    fun reduceDims(shape : Shape<TInt32>, axis : Operand<TInt32>) : Output<TInt32> {
        val op = shapeOps.reduceDims(shape, axis)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> reduceDims(operand : Operand<T>, axis : Operand<U>, dType : DataType<U>) : Output<T> {
        val op = shapeOps.reduceDims(operand, axis, dType)
        return op.asOutput()
    }

    fun <U : TNumber> reduceDims(shape : Shape<U>, axis : Operand<U>, dType : DataType<U>) : Output<U> {
        val op = shapeOps.reduceDims(shape, axis, dType)
        return op.asOutput()
    }

    fun size(shape : Shape<TInt32>) : Output<TInt32> {
        val op = shapeOps.size(shape)
        return op.asOutput()
    }

    fun <T : TType> size(input : Operand<T>, dim : Operand<TInt32>) : Output<TInt32> {
        val op = shapeOps.size(input, dim)
        return op.asOutput()
    }

    fun <U : TNumber> size(shape : Shape<U>, dType : DataType<U>) : Output<U> {
        val op = shapeOps.size(shape, dType)
        return op.asOutput()
    }

    fun size(shape : Shape<TInt32>, dim : Operand<TInt32>) : Output<TInt32> {
        val op = shapeOps.size(shape, dim)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> size(input : Operand<T>, dim : Operand<U>, dType : DataType<U>) : Output<U> {
        val op = shapeOps.size(input, dim, dType)
        return op.asOutput()
    }

    fun <U : TNumber> size(shape : Shape<U>, dim : Operand<U>, dType : DataType<U>) : Output<U> {
        val op = shapeOps.size(shape, dim, dType)
        return op.asOutput()
    }

    fun squeeze(shape : Shape<TInt32>) : Output<TInt32> {
        val op = shapeOps.squeeze(shape)
        return op.asOutput()
    }

    fun <U : TNumber> squeeze(shape : Shape<U>, dType : DataType<U>) : Output<U> {
        val op = shapeOps.squeeze(shape, dType)
        return op.asOutput()
    }

    fun tail(shape : Shape<TInt32>) : Output<TInt32> {
        val op = shapeOps.tail(shape)
        return op.asOutput()
    }

    fun <U : TNumber> tail(shape : Shape<U>, dType : DataType<U>) : Output<U> {
        val op = shapeOps.tail(shape, dType)
        return op.asOutput()
    }

    fun take(shape : Shape<TInt32>, n : Operand<TInt32>) : Output<TInt32> {
        val op = shapeOps.take(shape, n)
        return op.asOutput()
    }

    fun <U : TNumber> take(shape : Shape<U>, n : Operand<U>, dType : DataType<U>) : Output<U> {
        val op = shapeOps.take(shape, n, dType)
        return op.asOutput()
    }

    fun takeLast(shape : Shape<TInt32>, n : Operand<TInt32>) : Output<TInt32> {
        val op = shapeOps.takeLast<TInt32>(shape, n)
        return op.asOutput()
    }

    fun <U : TNumber> takeLast(shape : Shape<U>, n : Operand<U>, dType : DataType<U>) : Output<U> {
        val op = shapeOps.takeLast(shape, n, dType)
        return op.asOutput()
    }
}