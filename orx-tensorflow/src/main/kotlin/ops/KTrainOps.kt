package org.openrndr.extra.tensorflow.ops

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.TrainOps
import org.tensorflow.op.train.*
import org.tensorflow.types.TInt32
import org.tensorflow.types.TInt64
import org.tensorflow.types.TString
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

interface KTrainOps {
    val trainOps: TrainOps

    fun accumulatorNumAccumulated(handle : Operand<TString>) : Output<TInt32> {
        val op = trainOps.accumulatorNumAccumulated(handle)
        return op.asOutput()
    }

    fun <T : TType> accumulatorTakeGradient(handle : Operand<TString>, numRequired : Operand<TInt32>, dtype : DataType<T>) : Output<T> {
        val op = trainOps.accumulatorTakeGradient(handle, numRequired, dtype)
        return op.asOutput()
    }

    fun <T : TType> applyAdadelta(variable : Operand<T>, accum : Operand<T>, accumUpdate : Operand<T>, lr : Operand<T>, rho : Operand<T>, epsilon : Operand<T>, grad : Operand<T>, options : ApplyAdadelta.Options) : Output<T> {
        val op = trainOps.applyAdadelta(variable, accum, accumUpdate, lr, rho, epsilon, grad, options)
        return op.asOutput()
    }

    fun <T : TType> applyAdagrad(variable : Operand<T>, accum : Operand<T>, lr : Operand<T>, grad : Operand<T>, options : ApplyAdagrad.Options) : Output<T> {
        val op = trainOps.applyAdagrad(variable, accum, lr, grad, options)
        return op.asOutput()
    }

    fun <T : TType> applyAdagradDa(variable : Operand<T>, gradientAccumulator : Operand<T>, gradientSquaredAccumulator : Operand<T>, grad : Operand<T>, lr : Operand<T>, l1 : Operand<T>, l2 : Operand<T>, globalStep : Operand<TInt64>, options : ApplyAdagradDa.Options) : Output<T> {
        val op = trainOps.applyAdagradDa(variable, gradientAccumulator, gradientSquaredAccumulator, grad, lr, l1, l2, globalStep, options)
        return op.asOutput()
    }

    fun <T : TType> applyAdam(variable : Operand<T>, m : Operand<T>, v : Operand<T>, beta1Power : Operand<T>, beta2Power : Operand<T>, lr : Operand<T>, beta1 : Operand<T>, beta2 : Operand<T>, epsilon : Operand<T>, grad : Operand<T>, options : ApplyAdam.Options) : Output<T> {
        val op = trainOps.applyAdam(variable, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad, options)
        return op.asOutput()
    }

    fun <T : TType> applyAddSign(variable : Operand<T>, m : Operand<T>, lr : Operand<T>, alpha : Operand<T>, signDecay : Operand<T>, beta : Operand<T>, grad : Operand<T>, options : ApplyAddSign.Options) : Output<T> {
        val op = trainOps.applyAddSign(variable, m, lr, alpha, signDecay, beta, grad, options)
        return op.asOutput()
    }

    fun <T : TType> applyCenteredRmsProp(variable : Operand<T>, mg : Operand<T>, ms : Operand<T>, mom : Operand<T>, lr : Operand<T>, rho : Operand<T>, momentum : Operand<T>, epsilon : Operand<T>, grad : Operand<T>, options : ApplyCenteredRmsProp.Options) : Output<T> {
        val op = trainOps.applyCenteredRmsProp(variable, mg, ms, mom, lr, rho, momentum, epsilon, grad, options)
        return op.asOutput()
    }

    fun <T : TType> applyFtrl(variable : Operand<T>, accum : Operand<T>, linear : Operand<T>, grad : Operand<T>, lr : Operand<T>, l1 : Operand<T>, l2 : Operand<T>, l2Shrinkage : Operand<T>, lrPower : Operand<T>, options : ApplyFtrl.Options) : Output<T> {
        val op = trainOps.applyFtrl(variable, accum, linear, grad, lr, l1, l2, l2Shrinkage, lrPower, options)
        return op.asOutput()
    }

    fun <T : TType> applyGradientDescent(variable : Operand<T>, alpha : Operand<T>, delta : Operand<T>, options : ApplyGradientDescent.Options) : Output<T> {
        val op = trainOps.applyGradientDescent(variable, alpha, delta, options)
        return op.asOutput()
    }

    fun <T : TType> applyMomentum(variable : Operand<T>, accum : Operand<T>, lr : Operand<T>, grad : Operand<T>, momentum : Operand<T>, options : ApplyMomentum.Options) : Output<T> {
        val op = trainOps.applyMomentum(variable, accum, lr, grad, momentum, options)
        return op.asOutput()
    }

    fun <T : TType> applyPowerSign(variable : Operand<T>, m : Operand<T>, lr : Operand<T>, logbase : Operand<T>, signDecay : Operand<T>, beta : Operand<T>, grad : Operand<T>, options : ApplyPowerSign.Options) : Output<T> {
        val op = trainOps.applyPowerSign(variable, m, lr, logbase, signDecay, beta, grad, options)
        return op.asOutput()
    }

    fun <T : TType> applyProximalAdagrad(variable : Operand<T>, accum : Operand<T>, lr : Operand<T>, l1 : Operand<T>, l2 : Operand<T>, grad : Operand<T>, options : ApplyProximalAdagrad.Options) : Output<T> {
        val op = trainOps.applyProximalAdagrad(variable, accum, lr, l1, l2, grad, options)
        return op.asOutput()
    }

    fun <T : TType> applyProximalGradientDescent(variable : Operand<T>, alpha : Operand<T>, l1 : Operand<T>, l2 : Operand<T>, delta : Operand<T>, options : ApplyProximalGradientDescent.Options) : Output<T> {
        val op = trainOps.applyProximalGradientDescent(variable, alpha, l1, l2, delta, options)
        return op.asOutput()
    }

    fun <T : TType> applyRmsProp(variable : Operand<T>, ms : Operand<T>, mom : Operand<T>, lr : Operand<T>, rho : Operand<T>, momentum : Operand<T>, epsilon : Operand<T>, grad : Operand<T>, options : ApplyRmsProp.Options) : Output<T> {
        val op = trainOps.applyRmsProp(variable, ms, mom, lr, rho, momentum, epsilon, grad, options)
        return op.asOutput()
    }

    fun <T : TType> batchMatMul(x : Operand<T>, y : Operand<T>, options : BatchMatMul.Options) : Output<T> {
        val op = trainOps.batchMatMul(x, y, options)
        return op.asOutput()
    }

    fun <T : TType> conditionalAccumulator(dtype : DataType<T>, shape : Shape, options : ConditionalAccumulator.Options) : Output<TString> {
        val op = trainOps.conditionalAccumulator(dtype, shape, options)
        return op.asOutput()
    }

    fun <T : TType> preventGradient(input : Operand<T>, options : PreventGradient.Options) : Output<T> {
        val op = trainOps.preventGradient(input, options)
        return op.asOutput()
    }

    fun <T : TType> restoreSlice(filePattern : Operand<TString>, tensorName : Operand<TString>, shapeAndSlice : Operand<TString>, dt : DataType<T>, options : RestoreSlice.Options) : Output<T> {
        val op = trainOps.restoreSlice(filePattern, tensorName, shapeAndSlice, dt, options)
        return op.asOutput()
    }

    fun sdcaFprint(input : Operand<TString>) : Output<TInt64> {
        val op = trainOps.sdcaFprint(input)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> sparseApplyAdadelta(variable : Operand<T>, accum : Operand<T>, accumUpdate : Operand<T>, lr : Operand<T>, rho : Operand<T>, epsilon : Operand<T>, grad : Operand<T>, indices : Operand<U>, options : SparseApplyAdadelta.Options) : Output<T> {
        val op = trainOps.sparseApplyAdadelta(variable, accum, accumUpdate, lr, rho, epsilon, grad, indices, options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> sparseApplyAdagradDa(variable : Operand<T>, gradientAccumulator : Operand<T>, gradientSquaredAccumulator : Operand<T>, grad : Operand<T>, indices : Operand<U>, lr : Operand<T>, l1 : Operand<T>, l2 : Operand<T>, globalStep : Operand<TInt64>, options : SparseApplyAdagradDa.Options) : Output<T> {
        val op = trainOps.sparseApplyAdagradDa(variable, gradientAccumulator, gradientSquaredAccumulator, grad, indices, lr, l1, l2, globalStep, options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> sparseApplyCenteredRmsProp(variable : Operand<T>, mg : Operand<T>, ms : Operand<T>, mom : Operand<T>, lr : Operand<T>, rho : Operand<T>, momentum : Operand<T>, epsilon : Operand<T>, grad : Operand<T>, indices : Operand<U>, options : SparseApplyCenteredRmsProp.Options) : Output<T> {
        val op = trainOps.sparseApplyCenteredRmsProp(variable, mg, ms, mom, lr, rho, momentum, epsilon, grad, indices, options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> sparseApplyFtrl(variable : Operand<T>, accum : Operand<T>, linear : Operand<T>, grad : Operand<T>, indices : Operand<U>, lr : Operand<T>, l1 : Operand<T>, l2 : Operand<T>, l2Shrinkage : Operand<T>, lrPower : Operand<T>, options : SparseApplyFtrl.Options) : Output<T> {
        val op = trainOps.sparseApplyFtrl(variable, accum, linear, grad, indices, lr, l1, l2, l2Shrinkage, lrPower, options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> sparseApplyMomentum(variable : Operand<T>, accum : Operand<T>, lr : Operand<T>, grad : Operand<T>, indices : Operand<U>, momentum : Operand<T>, options : SparseApplyMomentum.Options) : Output<T> {
        val op = trainOps.sparseApplyMomentum(variable, accum, lr, grad, indices, momentum, options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> sparseApplyProximalAdagrad(variable : Operand<T>, accum : Operand<T>, lr : Operand<T>, l1 : Operand<T>, l2 : Operand<T>, grad : Operand<T>, indices : Operand<U>, options : SparseApplyProximalAdagrad.Options) : Output<T> {
        val op = trainOps.sparseApplyProximalAdagrad(variable, accum, lr, l1, l2, grad, indices, options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> sparseApplyProximalGradientDescent(variable : Operand<T>, alpha : Operand<T>, l1 : Operand<T>, l2 : Operand<T>, grad : Operand<T>, indices : Operand<U>, options : SparseApplyProximalGradientDescent.Options) : Output<T> {
        val op = trainOps.sparseApplyProximalGradientDescent(variable, alpha, l1, l2, grad, indices, options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> sparseApplyRmsProp(variable : Operand<T>, ms : Operand<T>, mom : Operand<T>, lr : Operand<T>, rho : Operand<T>, momentum : Operand<T>, epsilon : Operand<T>, grad : Operand<T>, indices : Operand<U>, options : SparseApplyRmsProp.Options) : Output<T> {
        val op = trainOps.sparseApplyRmsProp(variable, ms, mom, lr, rho, momentum, epsilon, grad, indices, options)
        return op.asOutput()
    }

    fun <T : TType> tileGrad(input : Operand<T>, multiples : Operand<TInt32>) : Output<T> {
        val op = trainOps.tileGrad(input, multiples)
        return op.asOutput()
    }
}
