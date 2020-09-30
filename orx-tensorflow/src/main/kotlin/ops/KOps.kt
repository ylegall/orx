package org.openrndr.extra.tensorflow.ops

import org.openrndr.extra.tensorflow.arrays.*
import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.Tensor
import org.tensorflow.ndarray.*
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.*

import org.tensorflow.types.*
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType
import java.nio.charset.Charset


interface KOps {
    val ops: Ops
    fun <T : TNumber> all(input: Operand<TBool>, axis: Operand<T>, vararg options: All.Options): Output<TBool> {
        val op = ops.all(input, axis, *options)
        return op.asOutput()
    }

    fun <T : TNumber> any(input: Operand<TBool>, axis: Operand<T>, vararg options: org.tensorflow.op.core.Any.Options): Output<TBool> {
        val op = ops.any(input, axis, *options)
        return op.asOutput()
    }

    fun array(vararg data: Int): Output<TInt32> {
        val op = ops.array(*data)
        return op.asOutput()
    }

    fun array(vararg data: String): Output<TString> {
        val op = ops.array(*data)
        return op.asOutput()
    }

    fun array(vararg data: Boolean): Output<TBool> {
        val op = ops.array(*data)
        return op.asOutput()
    }

    fun array(vararg data: Long): Output<TInt64> {
        val op = ops.array(*data)
        return op.asOutput()
    }

    fun array(vararg data: Float): Output<TFloat32> {
        val op = ops.array(*data)
        return op.asOutput()
    }

    fun array(vararg data: Double): Output<TFloat64> {
        val op = ops.array(*data)
        return op.asOutput()
    }

    fun array(vararg data: Byte): Output<TUint8> {
        val op = ops.array(*data)
        return op.asOutput()
    }

    fun array(charset: Charset, vararg data: String): Output<TString> {
        val op = ops.array(charset, *data)
        return op.asOutput()
    }

    fun <T : TType> assign(ref: Operand<T>, value: Operand<T>, vararg options: Assign.Options): Output<T> {
        val op = ops.assign(ref, value, *options)
        return op.asOutput()
    }

    fun <T : TType> assignAdd(ref: Operand<T>, value: Operand<T>, vararg options: AssignAdd.Options): Output<T> {
        val op = ops.assignAdd(ref, value, *options)
        return op.asOutput()
    }

    fun <T : TType> assignSub(ref: Operand<T>, value: Operand<T>, vararg options: AssignSub.Options): Output<T> {
        val op = ops.assignSub(ref, value, *options)
        return op.asOutput()
    }

    fun barrier(componentTypes: List<DataType<*>>, vararg options: Barrier.Options): Output<TString> {
        val op = ops.barrier(componentTypes, *options)
        return op.asOutput()
    }

    fun barrierIncompleteSize(handle: Operand<TString>): Output<TInt32> {
        val op = ops.barrierIncompleteSize(handle)
        return op.asOutput()
    }

    fun barrierReadySize(handle: Operand<TString>): Output<TInt32> {
        val op = ops.barrierReadySize(handle)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> batchToSpace(input: Operand<T>, crops: Operand<U>, blockSize: Long): Output<T> {
        val op = ops.batchToSpace(input, crops, blockSize)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber, V : TNumber> batchToSpaceNd(input: Operand<T>, blockShape: Operand<U>, crops: Operand<V>): Output<T> {
        val op = ops.batchToSpaceNd(input, blockShape, crops)
        return op.asOutput()
    }

    fun <U : TType, T : TType> bitcast(input: Operand<T>, type: DataType<U>): Output<U> {
        val op = ops.bitcast(input, type)
        return op.asOutput()
    }

    fun <T : TNumber> broadcastDynamicShape(s0: Operand<T>, s1: Operand<T>): Output<T> {
        val op = ops.broadcastDynamicShape(s0, s1)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> broadcastTo(input: Operand<T>, shape: Operand<U>): Output<T> {
        val op = ops.broadcastTo(input, shape)
        return op.asOutput()
    }

    fun <T : TNumber> bucketize(input: Operand<T>, boundaries: List<Float>): Output<TInt32> {
        val op = ops.bucketize(input, boundaries)
        return op.asOutput()
    }

    fun <T : TType> clipByValue(t: Operand<T>, clipValueMin: Operand<T>, clipValueMax: Operand<T>): Output<T> {
        val op = ops.clipByValue(t, clipValueMin, clipValueMax)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> concat(values: Iterable<Operand<T>>, axis: Operand<U>): Output<T> {
        val op = ops.concat(values, axis)
        return op.asOutput()
    }

    fun constant(data: LongNdArray): Output<TInt64> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: IntArray): Output<TInt32> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: IntArray3D): Output<TInt32> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: Double): Output<TFloat64> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: LongArray4D): Output<TInt64> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: BooleanArray4D): Output<TBool> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: IntNdArray): Output<TInt32> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: DoubleNdArray): Output<TFloat64> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: IntArray4D): Output<TInt32> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: FloatArray6D): Output<TFloat32> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: Byte): Output<TUint8> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: BooleanArray3D): Output<TBool> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: FloatArray4D): Output<TFloat32> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: LongArray2D): Output<TInt64> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: ByteArray5D): Output<TUint8> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: BooleanNdArray): Output<TBool> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: FloatArray2D): Output<TFloat32> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: ByteNdArray): Output<TUint8> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: ByteArray2D): Output<TUint8> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: DoubleArray5D): Output<TFloat64> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: FloatArray3D): Output<TFloat32> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: ByteArray): Output<TUint8> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: FloatArray): Output<TFloat32> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: BooleanArray2D): Output<TBool> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: NdArray<String>): Output<TString> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: String): Output<TString> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: DoubleArray4D): Output<TFloat64> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: DoubleArray2D): Output<TFloat64> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: Int): Output<TInt32> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: ByteArray4D): Output<TUint8> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: IntArray6D): Output<TInt32> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: Long): Output<TInt64> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: Float): Output<TFloat32> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: FloatArray5D): Output<TFloat32> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: DoubleArray3D): Output<TFloat64> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: LongArray6D): Output<TInt64> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: LongArray5D): Output<TInt64> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: LongArray): Output<TInt64> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: BooleanArray): Output<TBool> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: ByteArray3D): Output<TUint8> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: ByteArray6D): Output<TUint8> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: IntArray2D): Output<TInt32> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: FloatNdArray): Output<TFloat32> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: IntArray5D): Output<TInt32> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: DoubleArray): Output<TFloat64> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: BooleanArray6D): Output<TBool> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: DoubleArray6D): Output<TFloat64> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: Boolean): Output<TBool> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: BooleanArray5D): Output<TBool> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(data: LongArray3D): Output<TInt64> {
        val op = ops.constant(data)
        return op.asOutput()
    }

    fun constant(shape: Shape): Output<TInt64> {
        val op = ops.constant(shape)
        return op.asOutput()
    }

    fun <T : TType> constant(tensor: Tensor<T>): Output<T> {
        val op = ops.constant(tensor)
        return op.asOutput()
    }

    fun constant(charset: Charset, data: Array<String>): Output<TString> {
        val op = ops.constant(charset, data)
        return op.asOutput()
    }

    fun constant(charset: Charset, data: String): Output<TString> {
        val op = ops.constant(charset, data)
        return op.asOutput()
    }

    fun constant(charset: Charset, data: NdArray<String>): Output<TString> {
        val op = ops.constant(charset, data)
        return op.asOutput()
    }

    fun constant(shape: Shape, data: FloatDataBuffer): Output<TFloat32> {
        val op = ops.constant(shape, data)
        return op.asOutput()
    }

    fun constant(shape: Shape, data: BooleanDataBuffer): Output<TBool> {
        val op = ops.constant(shape, data)
        return op.asOutput()
    }

    fun constant(shape: Shape, data: ByteDataBuffer): Output<TUint8> {
        val op = ops.constant(shape, data)
        return op.asOutput()
    }

    fun constant(shape: Shape, data: LongDataBuffer): Output<TInt64> {
        val op = ops.constant(shape, data)
        return op.asOutput()
    }

    fun constant(shape: Shape, data: DataBuffer<String>): Output<TString> {
        val op = ops.constant(shape, data)
        return op.asOutput()
    }

    fun constant(shape: Shape, data: DoubleDataBuffer): Output<TFloat64> {
        val op = ops.constant(shape, data)
        return op.asOutput()
    }

    fun constant(shape: Shape, data: IntDataBuffer): Output<TInt32> {
        val op = ops.constant(shape, data)
        return op.asOutput()
    }

    fun constant(charset: Charset, shape: Shape, data: DataBuffer<String>): Output<TString> {
        val op = ops.constant(charset, shape, data)
        return op.asOutput()
    }

    fun <T : TType> constant(type: DataType<T>, shape: Shape, data: ByteDataBuffer): Output<T> {
        val op = ops.constant(type, shape, data)
        return op.asOutput()
    }

    fun <T : TNumber> countUpTo(ref: Operand<T>, limit: Long): Output<T> {
        val op = ops.countUpTo(ref, limit)
        return op.asOutput()
    }

    fun <T : TType> deepCopy(x: Operand<T>): Output<T> {
        val op = ops.deepCopy(x)
        return op.asOutput()
    }

    fun <T : TType> destroyTemporaryVariable(ref: Operand<T>, varName: String): Output<T> {
        val op = ops.destroyTemporaryVariable(ref, varName)
        return op.asOutput()
    }

    fun <T : TType> dynamicStitch(indices: Iterable<Operand<TInt32>>, data: Iterable<Operand<T>>): Output<T> {
        val op = ops.dynamicStitch(indices, data)
        return op.asOutput()
    }

    fun <T : TType> editDistance(hypothesisIndices: Operand<TInt64>, hypothesisValues: Operand<T>, hypothesisShape: Operand<TInt64>, truthIndices: Operand<TInt64>, truthValues: Operand<T>, truthShape: Operand<TInt64>, vararg options: EditDistance.Options): Output<TFloat32> {
        val op = ops.editDistance(hypothesisIndices, hypothesisValues, hypothesisShape, truthIndices, truthValues, truthShape, *options)
        return op.asOutput()
    }

    fun <T : TType> empty(shape: Operand<TInt32>, dtype: DataType<T>, vararg options: Empty.Options): Output<T> {
        val op = ops.empty(shape, dtype, *options)
        return op.asOutput()
    }

    fun <T : TNumber, U : TType> emptyTensorList(elementShape: Operand<T>, maxNumElements: Operand<TInt32>, elementDtype: DataType<U>): Output<TType> {
        val op = ops.emptyTensorList(elementShape, maxNumElements, elementDtype)
        return op.asOutput()
    }

    fun <T : TType> ensureShape(input: Operand<T>, shape: Shape): Output<T> {
        val op = ops.ensureShape(input, shape)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> expandDims(input: Operand<T>, axis: Operand<U>): Output<T> {
        val op = ops.expandDims(input, axis)
        return op.asOutput()
    }

    fun <T : TNumber> extractVolumePatches(input: Operand<T>, ksizes: List<Long>, strides: List<Long>, padding: String): Output<T> {
        val op = ops.extractVolumePatches(input, ksizes, strides, padding)
        return op.asOutput()
    }

    fun <U : TType, T : TNumber> fill(dims: Operand<T>, value: Operand<U>): Output<U> {
        val op = ops.fill(dims, value)
        return op.asOutput()
    }

    fun <T : TType> fingerprint(data: Operand<T>, method: Operand<TString>): Output<TUint8> {
        val op = ops.fingerprint(data, method)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber, V : TNumber> gather(params: Operand<T>, indices: Operand<U>, axis: Operand<V>, vararg options: Gather.Options): Output<T> {
        val op = ops.gather(params, indices, axis, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> gatherNd(params: Operand<T>, indices: Operand<U>): Output<T> {
        val op = ops.gatherNd(params, indices)
        return op.asOutput()
    }

    fun <T : TType> getSessionHandle(value: Operand<T>): Output<TType> {
        val op = ops.getSessionHandle(value)
        return op.asOutput()
    }

    fun <T : TType> getSessionTensor(handle: Operand<TString>, dtype: DataType<T>): Output<T> {
        val op = ops.getSessionTensor(handle, dtype)
        return op.asOutput()
    }

    fun <T : TType> guaranteeConst(input: Operand<T>): Output<T> {
        val op = ops.guaranteeConst(input)
        return op.asOutput()
    }

    fun <T : TType, U : TType> hashTable(keyDtype: DataType<T>, valueDtype: DataType<U>, vararg options: HashTable.Options): Output<TType> {
        val op = ops.hashTable(keyDtype, valueDtype, *options)
        return op.asOutput()
    }

    fun <T : TNumber> histogramFixedWidth(values: Operand<T>, valueRange: Operand<T>, nbins: Operand<TInt32>): Output<TInt32> {
        val op = ops.histogramFixedWidth(values, valueRange, nbins)
        return op.asOutput()
    }

    fun <U : TNumber, T : TNumber> histogramFixedWidth(values: Operand<T>, valueRange: Operand<T>, nbins: Operand<TInt32>, dtype: DataType<U>): Output<U> {
        val op = ops.histogramFixedWidth(values, valueRange, nbins, dtype)
        return op.asOutput()
    }

    fun <T : TType> identity(input: Operand<T>): Output<T> {
        val op = ops.identity(input)
        return op.asOutput()
    }

    fun <T : TType> immutableConst(dtype: DataType<T>, shape: Shape, memoryRegionName: String): Output<T> {
        val op = ops.immutableConst(dtype, shape, memoryRegionName)
        return op.asOutput()
    }
// omitted initAdd

    fun <T : TType> inplaceAdd(x: Operand<T>, i: Operand<TInt32>, v: Operand<T>): Output<T> {
        val op = ops.inplaceAdd(x, i, v)
        return op.asOutput()
    }

    fun <T : TType> inplaceSub(x: Operand<T>, i: Operand<TInt32>, v: Operand<T>): Output<T> {
        val op = ops.inplaceSub(x, i, v)
        return op.asOutput()
    }

    fun <T : TType> inplaceUpdate(x: Operand<T>, i: Operand<TInt32>, v: Operand<T>): Output<T> {
        val op = ops.inplaceUpdate(x, i, v)
        return op.asOutput()
    }

    fun <T : TType> isVariableInitialized(ref: Operand<T>): Output<TBool> {
        val op = ops.isVariableInitialized(ref)
        return op.asOutput()
    }

    fun <U : TType, T : TType> lookupTableFind(tableHandle: Operand<*>, keys: Operand<T>, defaultValue: Operand<U>): Output<U> {
        val op = ops.lookupTableFind(tableHandle, keys, defaultValue)
        return op.asOutput()
    }

    fun lookupTableSize(tableHandle: Operand<*>): Output<TInt64> {
        val op = ops.lookupTableSize(tableHandle)
        return op.asOutput()
    }

    fun loopCond(input: Operand<TBool>): Output<TBool> {
        val op = ops.loopCond(input)
        return op.asOutput()
    }

    fun mapIncompleteSize(dtypes: List<DataType<*>>, vararg options: MapIncompleteSize.Options): Output<TInt32> {
        val op = ops.mapIncompleteSize(dtypes, *options)
        return op.asOutput()
    }

    fun mapSize(dtypes: List<DataType<*>>, vararg options: MapSize.Options): Output<TInt32> {
        val op = ops.mapSize(dtypes, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> max(input: Operand<T>, axis: Operand<U>, vararg options: Max.Options): Output<T> {
        val op = ops.max(input, axis, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> min(input: Operand<T>, axis: Operand<U>, vararg options: Min.Options): Output<T> {
        val op = ops.min(input, axis, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> mirrorPad(input: Operand<T>, paddings: Operand<U>, mode: String): Output<T> {
        val op = ops.mirrorPad(input, paddings, mode)
        return op.asOutput()
    }

    fun <T : TType, U : TType> mutableDenseHashTable(emptyKey: Operand<T>, deletedKey: Operand<T>, valueDtype: DataType<U>, vararg options: MutableDenseHashTable.Options): Output<TType> {
        val op = ops.mutableDenseHashTable(emptyKey, deletedKey, valueDtype, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TType> mutableHashTable(keyDtype: DataType<T>, valueDtype: DataType<U>, vararg options: MutableHashTable.Options): Output<TType> {
        val op = ops.mutableHashTable(keyDtype, valueDtype, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TType> mutableHashTableOfTensors(keyDtype: DataType<T>, valueDtype: DataType<U>, vararg options: MutableHashTableOfTensors.Options): Output<TType> {
        val op = ops.mutableHashTableOfTensors(keyDtype, valueDtype, *options)
        return op.asOutput()
    }

    fun mutex(vararg options: Mutex.Options): Output<TType> {
        val op = ops.mutex(*options)
        return op.asOutput()
    }

    fun mutexLock(mutex: Operand<*>): Output<TType> {
        val op = ops.mutexLock(mutex)
        return op.asOutput()
    }

    fun <T : TType> nextIteration(data: Operand<T>): Output<T> {
        val op = ops.nextIteration(data)
        return op.asOutput()
    }

    fun <U : TType, T : TNumber> oneHot(indices: Operand<T>, depth: Operand<TInt32>, onValue: Operand<U>, offValue: Operand<U>, vararg options: OneHot.Options): Output<U> {
        val op = ops.oneHot(indices, depth, onValue, offValue, *options)
        return op.asOutput()
    }

    fun <T : TType> onesLike(x: Operand<T>): Output<T> {
        val op = ops.onesLike(x)
        return op.asOutput()
    }

    fun orderedMapIncompleteSize(dtypes: List<DataType<*>>, vararg options: OrderedMapIncompleteSize.Options): Output<TInt32> {
        val op = ops.orderedMapIncompleteSize(dtypes, *options)
        return op.asOutput()
    }

    fun orderedMapSize(dtypes: List<DataType<*>>, vararg options: OrderedMapSize.Options): Output<TInt32> {
        val op = ops.orderedMapSize(dtypes, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> pad(input: Operand<T>, paddings: Operand<U>, constantValues: Operand<T>): Output<T> {
        val op = ops.pad(input, paddings, constantValues)
        return op.asOutput()
    }

    fun <T : TType> parallelConcat(values: Iterable<Operand<T>>, shape: Shape): Output<T> {
        val op = ops.parallelConcat(values, shape)
        return op.asOutput()
    }

    fun <T : TType> parallelDynamicStitch(indices: Iterable<Operand<TInt32>>, data: Iterable<Operand<T>>): Output<T> {
        val op = ops.parallelDynamicStitch(indices, data)
        return op.asOutput()
    }

    fun <T : TType> placeholder(dtype: DataType<T>, vararg options: Placeholder.Options): Output<T> {
        val op = ops.placeholder(dtype, *options)
        return op.asOutput()
    }

    fun <T : TType> placeholderWithDefault(input: Operand<T>, shape: Shape): Output<T> {
        val op = ops.placeholderWithDefault(input, shape)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> prod(input: Operand<T>, axis: Operand<U>, vararg options: Prod.Options): Output<T> {
        val op = ops.prod(input, axis, *options)
        return op.asOutput()
    }

    fun <T : TNumber> range(start: Operand<T>, limit: Operand<T>, delta: Operand<T>): Output<T> {
        val op = ops.range(start, limit, delta)
        return op.asOutput()
    }

    fun <T : TType> rank(input: Operand<T>): Output<TInt32> {
        val op = ops.rank(input)
        return op.asOutput()
    }

    fun <T : TType> readVariableOp(resource: Operand<*>, dtype: DataType<T>): Output<T> {
        val op = ops.readVariableOp(resource, dtype)
        return op.asOutput()
    }

    fun <T : TNumber> reduceAll(input: Operand<TBool>, axis: Operand<T>, vararg options: ReduceAll.Options): Output<TBool> {
        val op = ops.reduceAll(input, axis, *options)
        return op.asOutput()
    }

    fun <T : TNumber> reduceAny(input: Operand<TBool>, axis: Operand<T>, vararg options: ReduceAny.Options): Output<TBool> {
        val op = ops.reduceAny(input, axis, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> reduceMax(input: Operand<T>, axis: Operand<U>, vararg options: ReduceMax.Options): Output<T> {
        val op = ops.reduceMax(input, axis, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> reduceMin(input: Operand<T>, axis: Operand<U>, vararg options: ReduceMin.Options): Output<T> {
        val op = ops.reduceMin(input, axis, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> reduceProd(input: Operand<T>, axis: Operand<U>, vararg options: ReduceProd.Options): Output<T> {
        val op = ops.reduceProd(input, axis, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> reduceSum(input: Operand<T>, axis: Operand<U>, vararg options: ReduceSum.Options): Output<T> {
        val op = ops.reduceSum(input, axis, *options)
        return op.asOutput()
    }

    fun <T : TType> refNextIteration(data: Operand<T>): Output<T> {
        val op = ops.refNextIteration(data)
        return op.asOutput()
    }

    fun <T : TType> refSelect(index: Operand<TInt32>, inputs: Iterable<Operand<T>>): Output<T> {
        val op = ops.refSelect(index, inputs)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> reshape(tensor: Operand<T>, shape: Operand<U>): Output<T> {
        val op = ops.reshape(tensor, shape)
        return op.asOutput()
    }

    fun <T : TNumber> resourceCountUpTo(resource: Operand<*>, limit: Long, t: DataType<T>): Output<T> {
        val op = ops.resourceCountUpTo(resource, limit, t)
        return op.asOutput()
    }

    fun <U : TType, T : TNumber> resourceGather(resource: Operand<*>, indices: Operand<T>, dtype: DataType<U>, vararg options: ResourceGather.Options): Output<U> {
        val op = ops.resourceGather(resource, indices, dtype, *options)
        return op.asOutput()
    }

    fun <U : TType, T : TNumber> resourceGatherNd(resource: Operand<*>, indices: Operand<T>, dtype: DataType<U>): Output<U> {
        val op = ops.resourceGatherNd(resource, indices, dtype)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> reverse(tensor: Operand<T>, axis: Operand<U>): Output<T> {
        val op = ops.reverse(tensor, axis)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> reverseSequence(input: Operand<T>, seqLengths: Operand<U>, seqDim: Long, vararg options: ReverseSequence.Options): Output<T> {
        val op = ops.reverseSequence(input, seqLengths, seqDim, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber, V : TNumber> roll(input: Operand<T>, shift: Operand<U>, axis: Operand<V>): Output<T> {
        val op = ops.roll(input, shift, axis)
        return op.asOutput()
    }

    fun rpc(address: Operand<TString>, method: Operand<TString>, request: Operand<TString>, vararg options: Rpc.Options): Output<TString> {
        val op = ops.rpc(address, method, request, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> scatterAdd(ref: Operand<T>, indices: Operand<U>, updates: Operand<T>, vararg options: ScatterAdd.Options): Output<T> {
        val op = ops.scatterAdd(ref, indices, updates, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> scatterDiv(ref: Operand<T>, indices: Operand<U>, updates: Operand<T>, vararg options: ScatterDiv.Options): Output<T> {
        val op = ops.scatterDiv(ref, indices, updates, *options)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> scatterMax(ref: Operand<T>, indices: Operand<U>, updates: Operand<T>, vararg options: ScatterMax.Options): Output<T> {
        val op = ops.scatterMax(ref, indices, updates, *options)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> scatterMin(ref: Operand<T>, indices: Operand<U>, updates: Operand<T>, vararg options: ScatterMin.Options): Output<T> {
        val op = ops.scatterMin(ref, indices, updates, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> scatterMul(ref: Operand<T>, indices: Operand<U>, updates: Operand<T>, vararg options: ScatterMul.Options): Output<T> {
        val op = ops.scatterMul(ref, indices, updates, *options)
        return op.asOutput()
    }

    fun <U : TType, T : TNumber> scatterNd(indices: Operand<T>, updates: Operand<U>, shape: Operand<T>): Output<U> {
        val op = ops.scatterNd(indices, updates, shape)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> scatterNdAdd(ref: Operand<T>, indices: Operand<U>, updates: Operand<T>, vararg options: ScatterNdAdd.Options): Output<T> {
        val op = ops.scatterNdAdd(ref, indices, updates, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> scatterNdNonAliasingAdd(input: Operand<T>, indices: Operand<U>, updates: Operand<T>): Output<T> {
        val op = ops.scatterNdNonAliasingAdd(input, indices, updates)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> scatterNdSub(ref: Operand<T>, indices: Operand<U>, updates: Operand<T>, vararg options: ScatterNdSub.Options): Output<T> {
        val op = ops.scatterNdSub(ref, indices, updates, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> scatterNdUpdate(ref: Operand<T>, indices: Operand<U>, updates: Operand<T>, vararg options: ScatterNdUpdate.Options): Output<T> {
        val op = ops.scatterNdUpdate(ref, indices, updates, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> scatterSub(ref: Operand<T>, indices: Operand<U>, updates: Operand<T>, vararg options: ScatterSub.Options): Output<T> {
        val op = ops.scatterSub(ref, indices, updates, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> scatterUpdate(ref: Operand<T>, indices: Operand<U>, updates: Operand<T>, vararg options: ScatterUpdate.Options): Output<T> {
        val op = ops.scatterUpdate(ref, indices, updates, *options)
        return op.asOutput()
    }

    fun <T : TType> select(condition: Operand<TBool>, t: Operand<T>, e: Operand<T>): Output<T> {
        val op = ops.select(condition, t, e)
        return op.asOutput()
    }

    fun <T : TType> setSize(setIndices: Operand<TInt64>, setValues: Operand<T>, setShape: Operand<TInt64>, vararg options: SetSize.Options): Output<TInt32> {
        val op = ops.setSize(setIndices, setValues, setShape, *options)
        return op.asOutput()
    }

    fun <T : TType> shape(input: Operand<T>): Output<TInt32> {
        val op = ops.shape(input)
        return op.asOutput()
    }

    fun <U : TNumber, T : TType> shape(input: Operand<T>, outType: DataType<U>): Output<U> {
        val op = ops.shape(input, outType)
        return op.asOutput()
    }

    fun <T : TType> size(input: Operand<T>): Output<TInt32> {
        val op = ops.size(input)
        return op.asOutput()
    }

    fun <U : TNumber, T : TType> size(input: Operand<T>, outType: DataType<U>): Output<U> {
        val op = ops.size(input, outType)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> slice(input: Operand<T>, begin: Operand<U>, size: Operand<U>): Output<T> {
        val op = ops.slice(input, begin, size)
        return op.asOutput()
    }

    fun <T : TType> snapshot(input: Operand<T>): Output<T> {
        val op = ops.snapshot(input)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber, V : TNumber> spaceToBatchNd(input: Operand<T>, blockShape: Operand<U>, paddings: Operand<V>): Output<T> {
        val op = ops.spaceToBatchNd(input, blockShape, paddings)
        return op.asOutput()
    }

    fun <T : TType> squeeze(input: Operand<T>, vararg options: Squeeze.Options): Output<T> {
        val op = ops.squeeze(input, *options)
        return op.asOutput()
    }

    fun <T : TType> stack(values: Iterable<Operand<T>>, vararg options: Stack.Options): Output<T> {
        val op = ops.stack(values, *options)
        return op.asOutput()
    }

    fun stageSize(dtypes: List<DataType<*>>, vararg options: StageSize.Options): Output<TInt32> {
        val op = ops.stageSize(dtypes, *options)
        return op.asOutput()
    }

    fun <T : TType> stopGradient(input: Operand<T>): Output<T> {
        val op = ops.stopGradient(input)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> stridedSlice(input: Operand<T>, begin: Operand<U>, end: Operand<U>, strides: Operand<U>, vararg options: StridedSlice.Options): Output<T> {
        val op = ops.stridedSlice(input, begin, end, strides, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> stridedSliceAssign(ref: Operand<T>, begin: Operand<U>, end: Operand<U>, strides: Operand<U>, value: Operand<T>, vararg options: StridedSliceAssign.Options): Output<T> {
        val op = ops.stridedSliceAssign(ref, begin, end, strides, value, *options)
        return op.asOutput()
    }

    fun <U : TType, T : TNumber> stridedSliceGrad(shape: Operand<T>, begin: Operand<T>, end: Operand<T>, strides: Operand<T>, dy: Operand<U>, vararg options: StridedSliceGrad.Options): Output<U> {
        val op = ops.stridedSliceGrad(shape, begin, end, strides, dy, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> sum(input: Operand<T>, axis: Operand<U>, vararg options: Sum.Options): Output<T> {
        val op = ops.sum(input, axis, *options)
        return op.asOutput()
    }

    fun <T : TType> temporaryVariable(shape: Shape, dtype: DataType<T>, vararg options: TemporaryVariable.Options): Output<T> {
        val op = ops.temporaryVariable(shape, dtype, *options)
        return op.asOutput()
    }

    fun <T : TType> tensorArrayGather(handle: Operand<*>, indices: Operand<TInt32>, flowIn: Operand<TFloat32>, dtype: DataType<T>, vararg options: TensorArrayGather.Options): Output<T> {
        val op = ops.tensorArrayGather(handle, indices, flowIn, dtype, *options)
        return op.asOutput()
    }

    fun <T : TType> tensorArrayPack(handle: Operand<TString>, flowIn: Operand<TFloat32>, dtype: DataType<T>, vararg options: TensorArrayPack.Options): Output<T> {
        val op = ops.tensorArrayPack(handle, flowIn, dtype, *options)
        return op.asOutput()
    }

    fun <T : TType> tensorArrayRead(handle: Operand<*>, index: Operand<TInt32>, flowIn: Operand<TFloat32>, dtype: DataType<T>): Output<T> {
        val op = ops.tensorArrayRead(handle, index, flowIn, dtype)
        return op.asOutput()
    }

    fun <T : TType> tensorArrayScatter(handle: Operand<*>, indices: Operand<TInt32>, value: Operand<T>, flowIn: Operand<TFloat32>): Output<TFloat32> {
        val op = ops.tensorArrayScatter(handle, indices, value, flowIn)
        return op.asOutput()
    }

    fun tensorArraySize(handle: Operand<*>, flowIn: Operand<TFloat32>): Output<TInt32> {
        val op = ops.tensorArraySize(handle, flowIn)
        return op.asOutput()
    }

    fun <T : TType> tensorArraySplit(handle: Operand<*>, value: Operand<T>, lengths: Operand<TInt64>, flowIn: Operand<TFloat32>): Output<TFloat32> {
        val op = ops.tensorArraySplit(handle, value, lengths, flowIn)
        return op.asOutput()
    }

    fun <T : TType> tensorArrayUnpack(handle: Operand<TString>, value: Operand<T>, flowIn: Operand<TFloat32>): Output<TFloat32> {
        val op = ops.tensorArrayUnpack(handle, value, flowIn)
        return op.asOutput()
    }

    fun <T : TType> tensorArrayWrite(handle: Operand<*>, index: Operand<TInt32>, value: Operand<T>, flowIn: Operand<TFloat32>): Output<TFloat32> {
        val op = ops.tensorArrayWrite(handle, index, value, flowIn)
        return op.asOutput()
    }

    fun <T : TType> tensorListConcatLists(inputA: Operand<*>, inputB: Operand<*>, elementDtype: DataType<T>): Output<TType> {
        val op = ops.tensorListConcatLists(inputA, inputB, elementDtype)
        return op.asOutput()
    }

    fun <T : TNumber> tensorListElementShape(inputHandle: Operand<*>, shapeType: DataType<T>): Output<T> {
        val op = ops.tensorListElementShape(inputHandle, shapeType)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> tensorListFromTensor(tensor: Operand<T>, elementShape: Operand<U>): Output<TType> {
        val op = ops.tensorListFromTensor(tensor, elementShape)
        return op.asOutput()
    }

    fun <T : TType> tensorListGather(inputHandle: Operand<*>, indices: Operand<TInt32>, elementShape: Operand<TInt32>, elementDtype: DataType<T>): Output<T> {
        val op = ops.tensorListGather(inputHandle, indices, elementShape, elementDtype)
        return op.asOutput()
    }

    fun <T : TType> tensorListGetItem(inputHandle: Operand<*>, index: Operand<TInt32>, elementShape: Operand<TInt32>, elementDtype: DataType<T>): Output<T> {
        val op = ops.tensorListGetItem(inputHandle, index, elementShape, elementDtype)
        return op.asOutput()
    }

    fun tensorListLength(inputHandle: Operand<*>): Output<TInt32> {
        val op = ops.tensorListLength(inputHandle)
        return op.asOutput()
    }

    fun <T : TType> tensorListPushBack(inputHandle: Operand<*>, tensor: Operand<T>): Output<TType> {
        val op = ops.tensorListPushBack(inputHandle, tensor)
        return op.asOutput()
    }

    fun <T : TType> tensorListPushBackBatch(inputHandles: Operand<*>, tensor: Operand<T>): Output<TType> {
        val op = ops.tensorListPushBackBatch(inputHandles, tensor)
        return op.asOutput()
    }

    fun <T : TNumber, U : TType> tensorListReserve(elementShape: Operand<T>, numElements: Operand<TInt32>, elementDtype: DataType<U>): Output<TType> {
        val op = ops.tensorListReserve(elementShape, numElements, elementDtype)
        return op.asOutput()
    }

    fun tensorListResize(inputHandle: Operand<*>, size: Operand<TInt32>): Output<TType> {
        val op = ops.tensorListResize(inputHandle, size)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> tensorListScatter(tensor: Operand<T>, indices: Operand<TInt32>, elementShape: Operand<U>, numElements: Operand<TInt32>): Output<TType> {
        val op = ops.tensorListScatter(tensor, indices, elementShape, numElements)
        return op.asOutput()
    }

    fun <T : TType> tensorListScatterIntoExistingList(inputHandle: Operand<*>, tensor: Operand<T>, indices: Operand<TInt32>): Output<TType> {
        val op = ops.tensorListScatterIntoExistingList(inputHandle, tensor, indices)
        return op.asOutput()
    }

    fun <T : TType> tensorListSetItem(inputHandle: Operand<*>, index: Operand<TInt32>, item: Operand<T>): Output<TType> {
        val op = ops.tensorListSetItem(inputHandle, index, item)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> tensorListSplit(tensor: Operand<T>, elementShape: Operand<U>, lengths: Operand<TInt64>): Output<TType> {
        val op = ops.tensorListSplit(tensor, elementShape, lengths)
        return op.asOutput()
    }

    fun <T : TType> tensorListStack(inputHandle: Operand<*>, elementShape: Operand<TInt32>, elementDtype: DataType<T>, vararg options: TensorListStack.Options): Output<T> {
        val op = ops.tensorListStack(inputHandle, elementShape, elementDtype, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> tensorScatterMax(tensor: Operand<T>, indices: Operand<U>, updates: Operand<T>): Output<T> {
        val op = ops.tensorScatterMax(tensor, indices, updates)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> tensorScatterMin(tensor: Operand<T>, indices: Operand<U>, updates: Operand<T>): Output<T> {
        val op = ops.tensorScatterMin(tensor, indices, updates)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> tensorScatterNdAdd(tensor: Operand<T>, indices: Operand<U>, updates: Operand<T>): Output<T> {
        val op = ops.tensorScatterNdAdd(tensor, indices, updates)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> tensorScatterNdMax(tensor: Operand<T>, indices: Operand<U>, updates: Operand<T>): Output<T> {
        val op = ops.tensorScatterNdMax(tensor, indices, updates)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> tensorScatterNdMin(tensor: Operand<T>, indices: Operand<U>, updates: Operand<T>): Output<T> {
        val op = ops.tensorScatterNdMin(tensor, indices, updates)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> tensorScatterNdSub(tensor: Operand<T>, indices: Operand<U>, updates: Operand<T>): Output<T> {
        val op = ops.tensorScatterNdSub(tensor, indices, updates)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> tensorScatterNdUpdate(tensor: Operand<T>, indices: Operand<U>, updates: Operand<T>): Output<T> {
        val op = ops.tensorScatterNdUpdate(tensor, indices, updates)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> tensorStridedSliceUpdate(input: Operand<T>, begin: Operand<U>, end: Operand<U>, strides: Operand<U>, value: Operand<T>, vararg options: TensorStridedSliceUpdate.Options): Output<T> {
        val op = ops.tensorStridedSliceUpdate(input, begin, end, strides, value, *options)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> tile(input: Operand<T>, multiples: Operand<U>): Output<T> {
        val op = ops.tile(input, multiples)
        return op.asOutput()
    }

    fun timestamp(): Output<TFloat64> {
        val op = ops.timestamp()
        return op.asOutput()
    }

    fun <T : TType> unbatch(batchedTensor: Operand<T>, batchIndex: Operand<TInt64>, id: Operand<TInt64>, timeoutMicros: Long, vararg options: Unbatch.Options): Output<T> {
        val op = ops.unbatch(batchedTensor, batchIndex, id, timeoutMicros, *options)
        return op.asOutput()
    }

    fun <T : TType> unbatchGrad(originalInput: Operand<T>, batchIndex: Operand<TInt64>, grad: Operand<T>, id: Operand<TInt64>, vararg options: UnbatchGrad.Options): Output<T> {
        val op = ops.unbatchGrad(originalInput, batchIndex, grad, id, *options)
        return op.asOutput()
    }

    fun <T : TNumber> unravelIndex(indices: Operand<T>, dims: Operand<T>): Output<T> {
        val op = ops.unravelIndex(indices, dims)
        return op.asOutput()
    }

    fun <T : TType> varHandleOp(dtype: DataType<T>, shape: Shape, vararg options: VarHandleOp.Options): Output<TType> {
        val op = ops.varHandleOp(dtype, shape, *options)
        return op.asOutput()
    }

    fun varIsInitializedOp(resource: Operand<*>): Output<TBool> {
        val op = ops.varIsInitializedOp(resource)
        return op.asOutput()
    }

    fun <T : TType> variable(init: Operand<T>, vararg options: Variable.Options): Output<T> {
        val op = ops.variable(init, *options)
        return op.asOutput()
    }

    fun <T : TType> variable(shape: Shape, dtype: DataType<T>, vararg options: Variable.Options): Output<T> {
        val op = ops.variable(shape, dtype, *options)
        return op.asOutput()
    }

    fun variableShape(input: Operand<*>): Output<TInt32> {
        val op = ops.variableShape(input)
        return op.asOutput()
    }

    fun <T : TNumber> variableShape(input: Operand<*>, outType: DataType<T>): Output<T> {
        val op = ops.variableShape(input, outType)
        return op.asOutput()
    }

    fun <T : TType> where(condition: Operand<T>): Output<TInt64> {
        val op = ops.where(condition)
        return op.asOutput()
    }

    fun <T : TType> xlaSpmdFullToShardShape(input: Operand<T>, manualSharding: String): Output<T> {
        val op = ops.xlaSpmdFullToShardShape(input, manualSharding)
        return op.asOutput()
    }

    fun <T : TType> xlaSpmdShardToFullShape(input: Operand<T>, manualSharding: String, fullShape: Shape): Output<T> {
        val op = ops.xlaSpmdShardToFullShape(input, manualSharding, fullShape)
        return op.asOutput()
    }

    fun <T : TType, U : TNumber> zeros(dims: Operand<U>, type: DataType<T>): Output<T> {
        val op = ops.zeros(dims, type)
        return op.asOutput()
    }

    fun <T : TType> zerosLike(x: Operand<T>): Output<T> {
        val op = ops.zerosLike(x)
        return op.asOutput()
    }

    fun shape(vararg dimensions: Long): Output<TInt64> {
        val s = Shape.of(*dimensions)
        return constant(s)
    }

// omitted withSubScope:Ops
// omitted withName:Ops
// omitted withControlDependencies:Ops
// omitted create:Ops
// omitted create:Ops
}
