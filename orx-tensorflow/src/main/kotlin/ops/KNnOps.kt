package org.openrndr.extra.tensorflow.ops

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.NnOps
import org.tensorflow.op.nn.*
import org.tensorflow.types.TBool
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

interface KNnOps {
    val nnOps: NnOps

    fun <T : TNumber> avgPool(value: Operand<T>, ksize: List<Long>, strides: List<Long>, padding: String, vararg options: AvgPool.Options): Output<T> {
        val op = nnOps.avgPool(value, ksize, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber> avgPool3d(input: Operand<T>, ksize: List<Long>, strides: List<Long>, padding: String, vararg options: AvgPool3d.Options): Output<T> {
        val op = nnOps.avgPool3d(input, ksize, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber> avgPool3dGrad(origInputShape: Operand<TInt32>, grad: Operand<T>, ksize: List<Long>, strides: List<Long>, padding: String, vararg options: AvgPool3dGrad.Options): Output<T> {
        val op = nnOps.avgPool3dGrad(origInputShape, grad, ksize, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TType> batchNormWithGlobalNormalization(t: Operand<T>, m: Operand<T>, v: Operand<T>, beta: Operand<T>, gamma: Operand<T>, varianceEpsilon: Float, scaleAfterNormalization: Boolean): Output<T> {
        val op = nnOps.batchNormWithGlobalNormalization(t, m, v, beta, gamma, varianceEpsilon, scaleAfterNormalization)
        return op.asOutput()
    }

    fun <T : TType> biasAdd(value: Operand<T>, bias: Operand<T>, vararg options: BiasAdd.Options): Output<T> {
        val op = nnOps.biasAdd(value, bias, *options)
        return op.asOutput()
    }

    fun <T : TType> biasAddGrad(outBackprop: Operand<T>, vararg options: BiasAddGrad.Options): Output<T> {
        val op = nnOps.biasAddGrad(outBackprop, *options)
        return op.asOutput()
    }

    fun <T : TNumber> conv2d(input: Operand<T>, filter: Operand<T>, strides: List<Long>, padding: String, vararg options: Conv2d.Options): Output<T> {
        val op = nnOps.conv2d(input, filter, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber> conv2dBackpropFilter(input: Operand<T>, filterSizes: Operand<TInt32>, outBackprop: Operand<T>, strides: List<Long>, padding: String, vararg options: Conv2dBackpropFilter.Options): Output<T> {
        val op = nnOps.conv2dBackpropFilter(input, filterSizes, outBackprop, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber> conv2dBackpropInput(inputSizes: Operand<TInt32>, filter: Operand<T>, outBackprop: Operand<T>, strides: List<Long>, padding: String, vararg options: Conv2dBackpropInput.Options): Output<T> {
        val op = nnOps.conv2dBackpropInput(inputSizes, filter, outBackprop, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber> conv3d(input: Operand<T>, filter: Operand<T>, strides: List<Long>, padding: String, vararg options: Conv3d.Options): Output<T> {
        val op = nnOps.conv3d(input, filter, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber> conv3dBackpropFilter(input: Operand<T>, filterSizes: Operand<TInt32>, outBackprop: Operand<T>, strides: List<Long>, padding: String, vararg options: Conv3dBackpropFilter.Options): Output<T> {
        val op = nnOps.conv3dBackpropFilter(input, filterSizes, outBackprop, strides, padding, *options)
        return op.asOutput()
    }

    fun <U : TNumber, T : TNumber> conv3dBackpropInput(inputSizes: Operand<T>, filter: Operand<U>, outBackprop: Operand<U>, strides: List<Long>, padding: String, vararg options: Conv3dBackpropInput.Options): Output<U> {
        val op = nnOps.conv3dBackpropInput(inputSizes, filter, outBackprop, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber> cudnnRNNCanonicalToParams(numLayers: Operand<TInt32>, numUnits: Operand<TInt32>, inputSize: Operand<TInt32>, weights: Iterable<Operand<T>>, biases: Iterable<Operand<T>>, vararg options: CudnnRNNCanonicalToParams.Options): Output<T> {
        val op = nnOps.cudnnRNNCanonicalToParams(numLayers, numUnits, inputSize, weights, biases, *options)
        return op.asOutput()
    }

    fun <U : TNumber, T : TNumber> cudnnRnnParamsSize(numLayers: Operand<TInt32>, numUnits: Operand<TInt32>, inputSize: Operand<TInt32>, t: DataType<T>, S: DataType<U>, vararg options: CudnnRnnParamsSize.Options): Output<U> {
        val op = nnOps.cudnnRnnParamsSize(numLayers, numUnits, inputSize, t, S, *options)
        return op.asOutput()
    }

    fun <T : TNumber> dataFormatDimMap(x: Operand<T>, vararg options: DataFormatDimMap.Options): Output<T> {
        val op = nnOps.dataFormatDimMap(x, *options)
        return op.asOutput()
    }

    fun <T : TNumber> dataFormatVecPermute(x: Operand<T>, vararg options: DataFormatVecPermute.Options): Output<T> {
        val op = nnOps.dataFormatVecPermute(x, *options)
        return op.asOutput()
    }

    fun <T : TType> depthToSpace(input: Operand<T>, blockSize: Long, vararg options: DepthToSpace.Options): Output<T> {
        val op = nnOps.depthToSpace(input, blockSize, *options)
        return op.asOutput()
    }

    fun <T : TNumber> depthwiseConv2dNative(input: Operand<T>, filter: Operand<T>, strides: List<Long>, padding: String, vararg options: DepthwiseConv2dNative.Options): Output<T> {
        val op = nnOps.depthwiseConv2dNative(input, filter, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber> depthwiseConv2dNativeBackpropFilter(input: Operand<T>, filterSizes: Operand<TInt32>, outBackprop: Operand<T>, strides: List<Long>, padding: String, vararg options: DepthwiseConv2dNativeBackpropFilter.Options): Output<T> {
        val op = nnOps.depthwiseConv2dNativeBackpropFilter(input, filterSizes, outBackprop, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber> depthwiseConv2dNativeBackpropInput(inputSizes: Operand<TInt32>, filter: Operand<T>, outBackprop: Operand<T>, strides: List<Long>, padding: String, vararg options: DepthwiseConv2dNativeBackpropInput.Options): Output<T> {
        val op = nnOps.depthwiseConv2dNativeBackpropInput(inputSizes, filter, outBackprop, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber> dilation2d(input: Operand<T>, filter: Operand<T>, strides: List<Long>, rates: List<Long>, padding: String): Output<T> {
        val op = nnOps.dilation2d(input, filter, strides, rates, padding)
        return op.asOutput()
    }

    fun <T : TNumber> dilation2dBackpropFilter(input: Operand<T>, filter: Operand<T>, outBackprop: Operand<T>, strides: List<Long>, rates: List<Long>, padding: String): Output<T> {
        val op = nnOps.dilation2dBackpropFilter(input, filter, outBackprop, strides, rates, padding)
        return op.asOutput()
    }

    fun <T : TNumber> dilation2dBackpropInput(input: Operand<T>, filter: Operand<T>, outBackprop: Operand<T>, strides: List<Long>, rates: List<Long>, padding: String): Output<T> {
        val op = nnOps.dilation2dBackpropInput(input, filter, outBackprop, strides, rates, padding)
        return op.asOutput()
    }

    fun <T : TNumber> elu(features: Operand<T>): Output<T> {
        val op = nnOps.elu(features)
        return op.asOutput()
    }

    fun <T : TNumber> fusedPadConv2d(input: Operand<T>, paddings: Operand<TInt32>, filter: Operand<T>, mode: String, strides: List<Long>, padding: String): Output<T> {
        val op = nnOps.fusedPadConv2d(input, paddings, filter, mode, strides, padding)
        return op.asOutput()
    }

    fun <T : TNumber> fusedResizeAndPadConv2d(input: Operand<T>, size: Operand<TInt32>, paddings: Operand<TInt32>, filter: Operand<T>, mode: String, strides: List<Long>, padding: String, vararg options: FusedResizeAndPadConv2d.Options): Output<T> {
        val op = nnOps.fusedResizeAndPadConv2d(input, size, paddings, filter, mode, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber> inTopK(predictions: Operand<TFloat32>, targets: Operand<T>, k: Operand<T>): Output<TBool> {
        val op = nnOps.inTopK(predictions, targets, k)
        return op.asOutput()
    }

    fun <T : TNumber> l2Loss(t: Operand<T>): Output<T> {
        val op = nnOps.l2Loss(t)
        return op.asOutput()
    }

    fun <T : TNumber> localResponseNormalization(input: Operand<T>, vararg options: LocalResponseNormalization.Options): Output<T> {
        val op = nnOps.localResponseNormalization(input, *options)
        return op.asOutput()
    }

    fun <T : TNumber> logSoftmax(logits: Operand<T>): Output<T> {
        val op = nnOps.logSoftmax(logits)
        return op.asOutput()
    }

    fun <T : TType> maxPool(input: Operand<T>, ksize: Operand<TInt32>, strides: Operand<TInt32>, padding: String, vararg options: MaxPool.Options): Output<T> {
        val op = nnOps.maxPool(input, ksize, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber> maxPool3d(input: Operand<T>, ksize: List<Long>, strides: List<Long>, padding: String, vararg options: MaxPool3d.Options): Output<T> {
        val op = nnOps.maxPool3d(input, ksize, strides, padding, *options)
        return op.asOutput()
    }

    fun <U : TNumber, T : TNumber> maxPool3dGrad(origInput: Operand<T>, origOutput: Operand<T>, grad: Operand<U>, ksize: List<Long>, strides: List<Long>, padding: String, vararg options: MaxPool3dGrad.Options): Output<U> {
        val op = nnOps.maxPool3dGrad(origInput, origOutput, grad, ksize, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber> maxPool3dGradGrad(origInput: Operand<T>, origOutput: Operand<T>, grad: Operand<T>, ksize: List<Long>, strides: List<Long>, padding: String, vararg options: MaxPool3dGradGrad.Options): Output<T> {
        val op = nnOps.maxPool3dGradGrad(origInput, origOutput, grad, ksize, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber> maxPoolGrad(origInput: Operand<T>, origOutput: Operand<T>, grad: Operand<T>, ksize: Operand<TInt32>, strides: Operand<TInt32>, padding: String, vararg options: MaxPoolGrad.Options): Output<T> {
        val op = nnOps.maxPoolGrad(origInput, origOutput, grad, ksize, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber> maxPoolGradGrad(origInput: Operand<T>, origOutput: Operand<T>, grad: Operand<T>, ksize: Operand<TInt32>, strides: Operand<TInt32>, padding: String, vararg options: MaxPoolGradGrad.Options): Output<T> {
        val op = nnOps.maxPoolGradGrad(origInput, origOutput, grad, ksize, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> maxPoolGradGradWithArgmax(input: Operand<T>, grad: Operand<T>, argmax: Operand<U>, ksize: List<Long>, strides: List<Long>, padding: String, vararg options: MaxPoolGradGradWithArgmax.Options): Output<T> {
        val op = nnOps.maxPoolGradGradWithArgmax(input, grad, argmax, ksize, strides, padding, *options)
        return op.asOutput()
    }

    fun <T : TNumber> nthElement(input: Operand<T>, n: Operand<TInt32>, vararg options: NthElement.Options): Output<T> {
        val op = nnOps.nthElement(input, n, *options)
        return op.asOutput()
    }

    fun <T : TType> relu(features: Operand<T>): Output<T> {
        val op = nnOps.relu(features)
        return op.asOutput()
    }

    fun <T : TNumber> relu6(features: Operand<T>): Output<T> {
        val op = nnOps.relu6(features)
        return op.asOutput()
    }

    fun <T : TNumber> selu(features: Operand<T>): Output<T> {
        val op = nnOps.selu(features)
        return op.asOutput()
    }

    fun <T : TNumber> sigmoidCrossEntropyWithLogits(labels: Operand<T>, logits: Operand<T>): Output<T> {
        val op = nnOps.sigmoidCrossEntropyWithLogits(labels, logits)
        return op.asOutput()
    }

    fun <T : TNumber> softmax(logits: Operand<T>): Output<T> {
        val op = nnOps.softmax(logits)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> softmaxCrossEntropyWithLogits(labels: Operand<U>, logits: Operand<T>, axis: Int): Output<T> {
        val op = nnOps.softmaxCrossEntropyWithLogits(labels, logits, axis)
        return op.asOutput()
    }

    fun <T : TNumber> softsign(features: Operand<T>): Output<T> {
        val op = nnOps.softsign(features)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> spaceToBatch(input: Operand<T>, paddings: Operand<U>, blockSize: Long): Output<T> {
        val op = nnOps.spaceToBatch(input, paddings, blockSize)
        return op.asOutput()
    }

    fun <T : TType> spaceToDepth(input: Operand<T>, blockSize: Long, vararg options: SpaceToDepth.Options): Output<T> {
        val op = nnOps.spaceToDepth(input, blockSize, *options)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> sparseSoftmaxCrossEntropyWithLogits(labels: Operand<T>, logits: Operand<U>): Output<*> {
        val op = nnOps.sparseSoftmaxCrossEntropyWithLogits(labels, logits)
        return op.asOutput()
    }
}
