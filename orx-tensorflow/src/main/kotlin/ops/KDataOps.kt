package org.openrndr.extra.tensorflow.ops

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.DataOps
import org.tensorflow.op.data.BatchDataset
import org.tensorflow.op.data.SerializeIterator
import org.tensorflow.types.TBool
import org.tensorflow.types.TInt64
import org.tensorflow.types.TString
import org.tensorflow.types.family.TType

interface KDataOps {
    val dataOps: DataOps

    fun batchDataset(inputDataset: Operand<*>, batchSize: Operand<TInt64>, dropRemainder: Operand<TBool>, outputTypes: List<DataType<*>>, outputShapes: List<Shape>, vararg options: BatchDataset.Options): Output<TType> {
        val op = dataOps.batchDataset(inputDataset, batchSize, dropRemainder, outputTypes, outputShapes, *options)
        return op.asOutput()
    }

    fun cSVDataset(filenames: Operand<TString>, compressionType: Operand<TString>, bufferSize: Operand<TInt64>, header: Operand<TBool>, fieldDelim: Operand<TString>, useQuoteDelim: Operand<TBool>, naValue: Operand<TString>, selectCols: Operand<TInt64>, recordDefaults: Iterable<Operand<*>>, outputShapes: List<Shape>): Output<TType> {
        val op = dataOps.cSVDataset(filenames, compressionType, bufferSize, header, fieldDelim, useQuoteDelim, naValue, selectCols, recordDefaults, outputShapes)
        return op.asOutput()
    }

    fun concatenateDataset(inputDataset: Operand<*>, anotherDataset: Operand<*>, outputTypes: List<DataType<*>>, outputShapes: List<Shape>): Output<TType> {
        val op = dataOps.concatenateDataset(inputDataset, anotherDataset, outputTypes, outputShapes)
        return op.asOutput()
    }

    fun iterator(sharedName: String, container: String, outputTypes: List<DataType<*>>, outputShapes: List<Shape>): Output<TType> {
        val op = dataOps.iterator(sharedName, container, outputTypes, outputShapes)
        return op.asOutput()
    }

    fun iteratorGetNextAsOptional(iterator: Operand<*>, outputTypes: List<DataType<*>>, outputShapes: List<Shape>): Output<TType> {
        val op = dataOps.iteratorGetNextAsOptional(iterator, outputTypes, outputShapes)
        return op.asOutput()
    }

    fun iteratorToStringHandle(resourceHandle: Operand<*>): Output<TString> {
        val op = dataOps.iteratorToStringHandle(resourceHandle)
        return op.asOutput()
    }

    fun optionalFromValue(components: Iterable<Operand<*>>): Output<TType> {
        val op = dataOps.optionalFromValue(components)
        return op.asOutput()
    }

    fun optionalHasValue(optional: Operand<*>): Output<TBool> {
        val op = dataOps.optionalHasValue(optional)
        return op.asOutput()
    }

    fun optionalNone(): Output<TType> {
        val op = dataOps.optionalNone()
        return op.asOutput()
    }

    fun rangeDataset(start: Operand<TInt64>, stop: Operand<TInt64>, step: Operand<TInt64>, outputTypes: List<DataType<*>>, outputShapes: List<Shape>): Output<TType> {
        val op = dataOps.rangeDataset(start, stop, step, outputTypes, outputShapes)
        return op.asOutput()
    }

    fun repeatDataset(inputDataset: Operand<*>, count: Operand<TInt64>, outputTypes: List<DataType<*>>, outputShapes: List<Shape>): Output<TType> {
        val op = dataOps.repeatDataset(inputDataset, count, outputTypes, outputShapes)
        return op.asOutput()
    }

    fun serializeIterator(resourceHandle: Operand<*>, vararg options: SerializeIterator.Options): Output<TType> {
        val op = dataOps.serializeIterator(resourceHandle, *options)
        return op.asOutput()
    }

    fun skipDataset(inputDataset: Operand<*>, count: Operand<TInt64>, outputTypes: List<DataType<*>>, outputShapes: List<Shape>): Output<TType> {
        val op = dataOps.skipDataset(inputDataset, count, outputTypes, outputShapes)
        return op.asOutput()
    }

    fun takeDataset(inputDataset: Operand<*>, count: Operand<TInt64>, outputTypes: List<DataType<*>>, outputShapes: List<Shape>): Output<TType> {
        val op = dataOps.takeDataset(inputDataset, count, outputTypes, outputShapes)
        return op.asOutput()
    }

    fun tensorSliceDataset(components: Iterable<Operand<*>>, outputShapes: List<Shape>): Output<TType> {
        val op = dataOps.tensorSliceDataset(components, outputShapes)
        return op.asOutput()
    }

    fun textLineDataset(filenames: Operand<TString>, compressionType: Operand<TString>, bufferSize: Operand<TInt64>): Output<TType> {
        val op = dataOps.textLineDataset(filenames, compressionType, bufferSize)
        return op.asOutput()
    }

    fun tfRecordDataset(filenames: Operand<TString>, compressionType: Operand<TString>, bufferSize: Operand<TInt64>): Output<TType> {
        val op = dataOps.tfRecordDataset(filenames, compressionType, bufferSize)
        return op.asOutput()
    }

    fun zipDataset(inputDatasets: Iterable<Operand<*>>, outputTypes: List<DataType<*>>, outputShapes: List<Shape>): Output<TType> {
        val op = dataOps.zipDataset(inputDatasets, outputTypes, outputShapes)
        return op.asOutput()
    }
}