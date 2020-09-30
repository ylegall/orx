package org.openrndr.extra.tensorflow.ops

import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.LinalgOps
import org.tensorflow.op.linalg.*
import org.tensorflow.types.*
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

interface KLinalgOps {
    val linalgOps: LinalgOps

    fun <T : TType, U : TNumber> bandPart(input: Operand<T>, numLower: Operand<U>, numUpper: Operand<U>): Output<T> {
        val op = linalgOps.bandPart(input, numLower, numUpper)
        return op.asOutput()
    }

    fun <T : TNumber> batchCholesky(input: Operand<T>): Output<T> {
        val op = linalgOps.batchCholesky(input)
        return op.asOutput()
    }

    fun <T : TNumber> batchCholeskyGrad(l: Operand<T>, grad: Operand<T>): Output<T> {
        val op = linalgOps.batchCholeskyGrad(l, grad)
        return op.asOutput()
    }

    fun <T : TType> batchMatrixBandPart(input: Operand<T>, numLower: Operand<TInt64>, numUpper: Operand<TInt64>): Output<T> {
        val op = linalgOps.batchMatrixBandPart(input, numLower, numUpper)
        return op.asOutput()
    }

    fun <T : TType> batchMatrixDeterminant(input: Operand<T>): Output<T> {
        val op = linalgOps.batchMatrixDeterminant(input)
        return op.asOutput()
    }

    fun <T : TType> batchMatrixDiag(diagonal: Operand<T>): Output<T> {
        val op = linalgOps.batchMatrixDiag(diagonal)
        return op.asOutput()
    }

    fun <T : TType> batchMatrixDiagPart(input: Operand<T>): Output<T> {
        val op = linalgOps.batchMatrixDiagPart(input)
        return op.asOutput()
    }

    fun <T : TNumber> batchMatrixInverse(input: Operand<T>, vararg options: BatchMatrixInverse.Options): Output<T> {
        val op = linalgOps.batchMatrixInverse(input, *options)
        return op.asOutput()
    }

    fun <T : TType> batchMatrixSetDiag(input: Operand<T>, diagonal: Operand<T>): Output<T> {
        val op = linalgOps.batchMatrixSetDiag(input, diagonal)
        return op.asOutput()
    }

    fun <T : TNumber> batchMatrixSolve(matrix: Operand<T>, rhs: Operand<T>, vararg options: BatchMatrixSolve.Options): Output<T> {
        val op = linalgOps.batchMatrixSolve(matrix, rhs, *options)
        return op.asOutput()
    }

    fun <T : TNumber> batchMatrixSolveLs(matrix: Operand<T>, rhs: Operand<T>, l2Regularizer: Operand<TFloat64>, vararg options: BatchMatrixSolveLs.Options): Output<T> {
        val op = linalgOps.batchMatrixSolveLs(matrix, rhs, l2Regularizer, *options)
        return op.asOutput()
    }

    fun <T : TNumber> batchMatrixTriangularSolve(matrix: Operand<T>, rhs: Operand<T>, vararg options: BatchMatrixTriangularSolve.Options): Output<T> {
        val op = linalgOps.batchMatrixTriangularSolve(matrix, rhs, *options)
        return op.asOutput()
    }

    fun <T : TType> cholesky(input: Operand<T>): Output<T> {
        val op = linalgOps.cholesky(input)
        return op.asOutput()
    }

    fun <T : TNumber> choleskyGrad(l: Operand<T>, grad: Operand<T>): Output<T> {
        val op = linalgOps.choleskyGrad(l, grad)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> conjugateTranspose(x: Operand<T>, perm: Operand<U>): Output<T> {
        val op = linalgOps.conjugateTranspose(x, perm)
        return op.asOutput()
    }

    fun <T : TNumber> cross(a: Operand<T>, b: Operand<T>): Output<T> {
        val op = linalgOps.cross(a, b)
        return op.asOutput()
    }

    fun <T : TType> det(input: Operand<T>): Output<T> {
        val op = linalgOps.det(input)
        return op.asOutput()
    }

    fun <T : TType> inv(input: Operand<T>, vararg options: Inv.Options): Output<T> {
        val op = linalgOps.inv(input, *options)
        return op.asOutput()
    }

    fun loadAndRemapMatrix(ckptPath: Operand<TString>, oldTensorName: Operand<TString>, rowRemapping: Operand<TInt64>, colRemapping: Operand<TInt64>, initializingValues: Operand<TFloat32>, numRows: Long, numCols: Long, vararg options: LoadAndRemapMatrix.Options): Output<TFloat32> {
        val op = linalgOps.loadAndRemapMatrix(ckptPath, oldTensorName, rowRemapping, colRemapping, initializingValues, numRows, numCols, *options)
        return op.asOutput()
    }

    fun <T : TType> matMul(a: Operand<T>, b: Operand<T>, vararg options: MatMul.Options): Output<T> {
        val op = linalgOps.matMul(a, b, *options)
        return op.asOutput()
    }

    fun <T : TType> matrixDiag(diagonal: Operand<T>, k: Operand<TInt32>, numRows: Operand<TInt32>, numCols: Operand<TInt32>, paddingValue: Operand<T>): Output<T> {
        val op = linalgOps.matrixDiag(diagonal, k, numRows, numCols, paddingValue)
        return op.asOutput()
    }

    fun <T : TType> matrixDiagPart(input: Operand<T>, k: Operand<TInt32>, paddingValue: Operand<T>): Output<T> {
        val op = linalgOps.matrixDiagPart(input, k, paddingValue)
        return op.asOutput()
    }

    fun <T : TType> matrixDiagPartV3(input: Operand<T>, k: Operand<TInt32>, paddingValue: Operand<T>, vararg options: MatrixDiagPartV3.Options): Output<T> {
        val op = linalgOps.matrixDiagPartV3(input, k, paddingValue, *options)
        return op.asOutput()
    }

    fun <T : TType> matrixDiagV3(diagonal: Operand<T>, k: Operand<TInt32>, numRows: Operand<TInt32>, numCols: Operand<TInt32>, paddingValue: Operand<T>, vararg options: MatrixDiagV3.Options): Output<T> {
        val op = linalgOps.matrixDiagV3(diagonal, k, numRows, numCols, paddingValue, *options)
        return op.asOutput()
    }

    fun <T : TType> matrixSetDiag(input: Operand<T>, diagonal: Operand<T>, k: Operand<TInt32>, vararg options: MatrixSetDiag.Options): Output<T> {
        val op = linalgOps.matrixSetDiag(input, diagonal, k, *options)
        return op.asOutput()
    }

    fun <T : TType> matrixSolveLs(matrix: Operand<T>, rhs: Operand<T>, l2Regularizer: Operand<TFloat64>, vararg options: MatrixSolveLs.Options): Output<T> {
        val op = linalgOps.matrixSolveLs(matrix, rhs, l2Regularizer, *options)
        return op.asOutput()
    }

    fun <T : TType> solve(matrix: Operand<T>, rhs: Operand<T>, vararg options: Solve.Options): Output<T> {
        val op = linalgOps.solve(matrix, rhs, *options)
        return op.asOutput()
    }

    fun <T : TType> sqrtm(input: Operand<T>): Output<T> {
        val op = linalgOps.sqrtm(input)
        return op.asOutput()
    }

    fun <T : TType> tensorDiag(diagonal: Operand<T>): Output<T> {
        val op = linalgOps.tensorDiag(diagonal)
        return op.asOutput()
    }

    fun <T : TType> tensorDiagPart(input: Operand<T>): Output<T> {
        val op = linalgOps.tensorDiagPart(input)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> transpose(x: Operand<T>, perm: Operand<U>): Output<T> {
        val op = linalgOps.transpose(x, perm)
        return op.asOutput()
    }

    fun <T : TType> triangularSolve(matrix: Operand<T>, rhs: Operand<T>, vararg options: TriangularSolve.Options): Output<T> {
        val op = linalgOps.triangularSolve(matrix, rhs, *options)
        return op.asOutput()
    }
}
