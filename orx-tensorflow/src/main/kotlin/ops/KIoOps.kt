package org.openrndr.extra.tensorflow.ops

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.IoOps
import org.tensorflow.op.io.*
import org.tensorflow.types.TBool
import org.tensorflow.types.TInt32
import org.tensorflow.types.TInt64
import org.tensorflow.types.TString
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

interface KIoOps {
    val ioOps: IoOps

    fun decodeBase64(input: Operand<TString>): Output<TString> {
        val op = ioOps.decodeBase64(input)
        return op.asOutput()
    }

    fun decodeCompressed(bytes: Operand<TString>, vararg options: DecodeCompressed.Options): Output<TString> {
        val op = ioOps.decodeCompressed(bytes, *options)
        return op.asOutput()
    }

    fun decodeJsonExample(jsonExamples: Operand<TString>): Output<TString> {
        val op = ioOps.decodeJsonExample(jsonExamples)
        return op.asOutput()
    }

    fun <T : TNumber> decodePaddedRaw(inputBytes: Operand<TString>, fixedLength: Operand<TInt32>, outType: DataType<T>, vararg options: DecodePaddedRaw.Options): Output<T> {
        val op = ioOps.decodePaddedRaw(inputBytes, fixedLength, outType, *options)
        return op.asOutput()
    }

    fun <T : TType> decodeRaw(bytes: Operand<TString>, outType: DataType<T>, vararg options: DecodeRaw.Options): Output<T> {
        val op = ioOps.decodeRaw(bytes, outType, *options)
        return op.asOutput()
    }

    fun encodeBase64(input: Operand<TString>, vararg options: EncodeBase64.Options): Output<TString> {
        val op = ioOps.encodeBase64(input, *options)
        return op.asOutput()
    }

    fun fifoQueue(componentTypes: List<DataType<*>>, vararg options: FifoQueue.Options): Output<TType> {
        val op = ioOps.fifoQueue(componentTypes, *options)
        return op.asOutput()
    }

    fun fixedLengthRecordReader(recordBytes: Long, vararg options: FixedLengthRecordReader.Options): Output<TType> {
        val op = ioOps.fixedLengthRecordReader(recordBytes, *options)
        return op.asOutput()
    }

    fun identityReader(vararg options: IdentityReader.Options): Output<TType> {
        val op = ioOps.identityReader(*options)
        return op.asOutput()
    }

    fun lmdbReader(vararg options: LmdbReader.Options): Output<TString> {
        val op = ioOps.lmdbReader(*options)
        return op.asOutput()
    }

    fun matchingFiles(pattern: Operand<TString>): Output<TString> {
        val op = ioOps.matchingFiles(pattern)
        return op.asOutput()
    }

    fun paddingFifoQueue(componentTypes: List<DataType<*>>, vararg options: PaddingFifoQueue.Options): Output<TType> {
        val op = ioOps.paddingFifoQueue(componentTypes, *options)
        return op.asOutput()
    }

    fun <T : TType> parseTensor(serialized: Operand<TString>, outType: DataType<T>): Output<T> {
        val op = ioOps.parseTensor(serialized, outType)
        return op.asOutput()
    }

    fun priorityQueue(componentTypes: List<DataType<*>>, shapes: List<Shape>, vararg options: PriorityQueue.Options): Output<TType> {
        val op = ioOps.priorityQueue(componentTypes, shapes, *options)
        return op.asOutput()
    }

    fun queueIsClosed(handle: Operand<*>): Output<TBool> {
        val op = ioOps.queueIsClosed(handle)
        return op.asOutput()
    }

    fun queueSize(handle: Operand<*>): Output<TInt32> {
        val op = ioOps.queueSize(handle)
        return op.asOutput()
    }

    fun randomShuffleQueue(componentTypes: List<DataType<*>>, vararg options: RandomShuffleQueue.Options): Output<TType> {
        val op = ioOps.randomShuffleQueue(componentTypes, *options)
        return op.asOutput()
    }

    fun readFile(filename: Operand<TString>): Output<TString> {
        val op = ioOps.readFile(filename)
        return op.asOutput()
    }

    fun readerNumRecordsProduced(readerHandle: Operand<*>): Output<TInt64> {
        val op = ioOps.readerNumRecordsProduced(readerHandle)
        return op.asOutput()
    }

    fun readerNumWorkUnitsCompleted(readerHandle: Operand<*>): Output<TInt64> {
        val op = ioOps.readerNumWorkUnitsCompleted(readerHandle)
        return op.asOutput()
    }

    fun readerSerializeState(readerHandle: Operand<*>): Output<TString> {
        val op = ioOps.readerSerializeState(readerHandle)
        return op.asOutput()
    }

    fun <T : TType> serializeManySparse(sparseIndices: Operand<TInt64>, sparseValues: Operand<T>, sparseShape: Operand<TInt64>): Output<TString>? {
        val op = ioOps.serializeManySparse(sparseIndices, sparseValues, sparseShape)
        return op.asOutput()
    }

    fun <U : TType, T : TType> serializeManySparse(sparseIndices: Operand<TInt64>, sparseValues: Operand<T>, sparseShape: Operand<TInt64>, outType: DataType<U>): Output<U> {
        val op = ioOps.serializeManySparse(sparseIndices, sparseValues, sparseShape, outType)
        return op.asOutput()
    }

    fun <T : TType> serializeSparse(sparseIndices: Operand<TInt64>, sparseValues: Operand<T>, sparseShape: Operand<TInt64>): Output<TString> {
        val op = ioOps.serializeSparse(sparseIndices, sparseValues, sparseShape)
        return op.asOutput()
    }

    fun <U : TType, T : TType> serializeSparse(sparseIndices: Operand<TInt64>, sparseValues: Operand<T>, sparseShape: Operand<TInt64>, outType: DataType<U>): Output<U> {
        val op = ioOps.serializeSparse(sparseIndices, sparseValues, sparseShape, outType)
        return op.asOutput()
    }

    fun <T : TType> serializeTensor(tensor: Operand<T>): Output<TString> {
        val op = ioOps.serializeTensor(tensor)
        return op.asOutput()
    }

    fun shardedFilename(basename: Operand<TString>, shard: Operand<TInt32>, numShards: Operand<TInt32>): Output<TString> {
        val op = ioOps.shardedFilename(basename, shard, numShards)
        return op.asOutput()
    }

    fun shardedFilespec(basename: Operand<TString>, numShards: Operand<TInt32>): Output<TString> {
        val op = ioOps.shardedFilespec(basename, numShards)
        return op.asOutput()
    }

    fun textLineReader(vararg options: TextLineReader.Options): Output<TType> {
        val op = ioOps.textLineReader(*options)
        return op.asOutput()
    }

    fun tfRecordReader(vararg options: TfRecordReader.Options): Output<TType> {
        val op = ioOps.tfRecordReader(*options)
        return op.asOutput()
    }

    fun wholeFileReader(vararg options: WholeFileReader.Options): Output<TType> {
        val op = ioOps.wholeFileReader(*options)
        return op.asOutput()
    }
}