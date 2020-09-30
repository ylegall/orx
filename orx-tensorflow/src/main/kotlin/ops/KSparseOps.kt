package org.openrndr.extra.tensorflow.ops

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.SparseOps
import org.tensorflow.op.sparse.*
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32
import org.tensorflow.types.TInt64
import org.tensorflow.types.TString
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

interface KSparseOps {
    val sparseOps: SparseOps

    fun <T : TType> addManySparseToTensorsMap(sparseIndices: Operand<TInt64>, sparseValues: Operand<T>, sparseShape: Operand<TInt64>, vararg options: AddManySparseToTensorsMap.Options): Output<TInt64> {
        val op = sparseOps.addManySparseToTensorsMap(sparseIndices, sparseValues, sparseShape, *options)
        return op.asOutput()
    }

    fun <T : TType> addSparseToTensorsMap(sparseIndices: Operand<TInt64>, sparseValues: Operand<T>, sparseShape: Operand<TInt64>, vararg options: AddSparseToTensorsMap.Options): Output<TInt64> {
        val op = sparseOps.addSparseToTensorsMap(sparseIndices, sparseValues, sparseShape, *options)
        return op.asOutput()
    }

    fun <U : TNumber, T : TNumber> sparseBincount(indices: Operand<TInt64>, values: Operand<T>, denseShape: Operand<TInt64>, size: Operand<T>, weights: Operand<U>, vararg options: SparseBincount.Options): Output<U> {
        val op = sparseOps.sparseBincount(indices, values, denseShape, size, weights, *options)
        return op.asOutput()
    }

    fun <T : TType> sparseConditionalAccumulator(dtype: DataType<T>, shape: Shape, vararg options: SparseConditionalAccumulator.Options): Output<TString> {
        val op = sparseOps.sparseConditionalAccumulator(dtype, shape, *options)
        return op.asOutput()
    }

    fun <T : TType> sparseDenseCwiseAdd(spIndices: Operand<TInt64>, spValues: Operand<T>, spShape: Operand<TInt64>, dense: Operand<T>): Output<T> {
        val op = sparseOps.sparseDenseCwiseAdd(spIndices, spValues, spShape, dense)
        return op.asOutput()
    }

    fun <T : TType> sparseDenseCwiseDiv(spIndices: Operand<TInt64>, spValues: Operand<T>, spShape: Operand<TInt64>, dense: Operand<T>): Output<T> {
        val op = sparseOps.sparseDenseCwiseDiv(spIndices, spValues, spShape, dense)
        return op.asOutput()
    }

    fun <T : TType> sparseDenseCwiseMul(spIndices: Operand<TInt64>, spValues: Operand<T>, spShape: Operand<TInt64>, dense: Operand<T>): Output<T> {
        val op = sparseOps.sparseDenseCwiseMul(spIndices, spValues, spShape, dense)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> sparseMatMul(a: Operand<T>, b: Operand<U>, vararg options: SparseMatMul.Options): Output<TFloat32> {
        val op = sparseOps.sparseMatMul(a, b, *options)
        return op.asOutput()
    }

    fun <T : TNumber> sparseReduceMax(inputIndices: Operand<TInt64>, inputValues: Operand<T>, inputShape: Operand<TInt64>, reductionAxes: Operand<TInt32>, vararg options: SparseReduceMax.Options): Output<T> {
        val op = sparseOps.sparseReduceMax(inputIndices, inputValues, inputShape, reductionAxes, *options)
        return op.asOutput()
    }

    fun <T : TType> sparseReduceSum(inputIndices: Operand<TInt64>, inputValues: Operand<T>, inputShape: Operand<TInt64>, reductionAxes: Operand<TInt32>, vararg options: SparseReduceSum.Options): Output<T> {
        val op = sparseOps.sparseReduceSum(inputIndices, inputValues, inputShape, reductionAxes, *options)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber, V : TNumber> sparseSegmentMean(data: Operand<T>, indices: Operand<U>, segmentIds: Operand<V>): Output<T> {
        val op = sparseOps.sparseSegmentMean(data, indices, segmentIds)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber, V : TNumber> sparseSegmentMeanGrad(grad: Operand<T>, indices: Operand<U>, segmentIds: Operand<V>, outputDim0: Operand<TInt32>): Output<T> {
        val op = sparseOps.sparseSegmentMeanGrad(grad, indices, segmentIds, outputDim0)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber, V : TNumber, W : TNumber> sparseSegmentMeanWithNumSegments(data: Operand<T>, indices: Operand<U>, segmentIds: Operand<V>, numSegments: Operand<W>): Output<T> {
        val op = sparseOps.sparseSegmentMeanWithNumSegments(data, indices, segmentIds, numSegments)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber, V : TNumber> sparseSegmentSqrtN(data: Operand<T>, indices: Operand<U>, segmentIds: Operand<V>): Output<T> {
        val op = sparseOps.sparseSegmentSqrtN(data, indices, segmentIds)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber, V : TNumber> sparseSegmentSqrtNGrad(grad: Operand<T>, indices: Operand<U>, segmentIds: Operand<V>, outputDim0: Operand<TInt32>): Output<T> {
        val op = sparseOps.sparseSegmentSqrtNGrad(grad, indices, segmentIds, outputDim0)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber, V : TNumber, W : TNumber> sparseSegmentSqrtNWithNumSegments(data: Operand<T>, indices: Operand<U>, segmentIds: Operand<V>, numSegments: Operand<W>): Output<T> {
        val op = sparseOps.sparseSegmentSqrtNWithNumSegments(data, indices, segmentIds, numSegments)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber, V : TNumber> sparseSegmentSum(data: Operand<T>, indices: Operand<U>, segmentIds: Operand<V>): Output<T> {
        val op = sparseOps.sparseSegmentSum(data, indices, segmentIds)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber, V : TNumber, W : TNumber> sparseSegmentSumWithNumSegments(data: Operand<T>, indices: Operand<U>, segmentIds: Operand<V>, numSegments: Operand<W>): Output<T> {
        val op = sparseOps.sparseSegmentSumWithNumSegments(data, indices, segmentIds, numSegments)
        return op.asOutput()
    }

    fun <T : TType> sparseSliceGrad(backpropValGrad: Operand<T>, inputIndices: Operand<TInt64>, inputStart: Operand<TInt64>, outputIndices: Operand<TInt64>): Output<T> {
        val op = sparseOps.sparseSliceGrad(backpropValGrad, inputIndices, inputStart, outputIndices)
        return op.asOutput()
    }

    fun <T : TNumber> sparseSoftmax(spIndices: Operand<TInt64>, spValues: Operand<T>, spShape: Operand<TInt64>): Output<T> {
        val op = sparseOps.sparseSoftmax(spIndices, spValues, spShape)
        return op.asOutput()
    }

    fun <U : TType, T : TNumber> sparseTensorDenseAdd(aIndices: Operand<T>, aValues: Operand<U>, aShape: Operand<T>, b: Operand<U>): Output<U> {
        val op = sparseOps.sparseTensorDenseAdd(aIndices, aValues, aShape, b)
        return op.asOutput()
    }

    fun <U : TType, T : TNumber> sparseTensorDenseMatMul(aIndices: Operand<T>, aValues: Operand<U>, aShape: Operand<TInt64>, b: Operand<U>, vararg options: SparseTensorDenseMatMul.Options): Output<U> {
        val op = sparseOps.sparseTensorDenseMatMul(aIndices, aValues, aShape, b, *options)
        return op.asOutput()
    }

    fun <U : TType, T : TNumber> sparseToDense(sparseIndices: Operand<T>, outputShape: Operand<T>, sparseValues: Operand<U>, defaultValue: Operand<U>, vararg options: SparseToDense.Options): Output<U> {
        val op = sparseOps.sparseToDense(sparseIndices, outputShape, sparseValues, defaultValue, *options)
        return op.asOutput()
    }
}
