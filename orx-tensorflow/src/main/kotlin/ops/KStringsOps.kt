package org.openrndr.extra.tensorflow.ops

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.StringsOps
import org.tensorflow.op.strings.*
import org.tensorflow.types.*
import org.tensorflow.types.family.TNumber

interface KStringsOps {
    val stringsOps: StringsOps

    fun join(inputs: Iterable<Operand<TString>>, vararg options: Join.Options): Output<TString> {
        val op = stringsOps.join(inputs, *options)
        return op.asOutput()
    }

    fun lower(input: Operand<TString>, vararg options: Lower.Options): Output<TString> {
        val op = stringsOps.lower(input, *options)
        return op.asOutput()
    }

    fun reduceJoin(inputs: Operand<TString>, reductionIndices: Operand<TInt32>, vararg options: ReduceJoin.Options): Output<TString> {
        val op = stringsOps.reduceJoin(inputs, reductionIndices, *options)
        return op.asOutput()
    }

    fun regexFullMatch(input: Operand<TString>, pattern: Operand<TString>): Output<TBool> {
        val op = stringsOps.regexFullMatch(input, pattern)
        return op.asOutput()
    }

    fun regexReplace(input: Operand<TString>, pattern: Operand<TString>, rewrite: Operand<TString>, vararg options: RegexReplace.Options): Output<TString> {
        val op = stringsOps.regexReplace(input, pattern, rewrite, *options)
        return op.asOutput()
    }

    fun stringFormat(inputs: Iterable<Operand<*>>, vararg options: StringFormat.Options): Output<TString> {
        val op = stringsOps.stringFormat(inputs, *options)
        return op.asOutput()
    }

    fun stringLength(input: Operand<TString>, vararg options: StringLength.Options): Output<TInt32> {
        val op = stringsOps.stringLength(input, *options)
        return op.asOutput()
    }

    fun strip(input: Operand<TString>): Output<TString> {
        val op = stringsOps.strip(input)
        return op.asOutput()
    }

    fun <T : TNumber> substr(input: Operand<TString>, pos: Operand<T>, len: Operand<T>, vararg options: Substr.Options): Output<TString> {
        val op = stringsOps.substr(input, pos, len, *options)
        return op.asOutput()
    }

    fun toHashBucket(stringTensor: Operand<TString>, numBuckets: Long): Output<TInt64> {
        val op = stringsOps.toHashBucket(stringTensor, numBuckets)
        return op.asOutput()
    }

    fun toHashBucketFast(input: Operand<TString>, numBuckets: Long): Output<TInt64> {
        val op = stringsOps.toHashBucketFast(input, numBuckets)
        return op.asOutput()
    }

    fun toHashBucketStrong(input: Operand<TString>, numBuckets: Long, key: List<Long>): Output<TInt64> {
        val op = stringsOps.toHashBucketStrong(input, numBuckets, key)
        return op.asOutput()
    }

    fun toNumber(stringTensor: Operand<TString>): Output<TFloat32> {
        val op = stringsOps.toNumber(stringTensor)
        return op.asOutput()
    }

    fun <T : TNumber> toNumber(stringTensor: Operand<TString>, outType: DataType<T>): Output<T> {
        val op = stringsOps.toNumber(stringTensor, outType)
        return op.asOutput()
    }

    fun unicodeScript(input: Operand<TInt32>): Output<TInt32> {
        val op = stringsOps.unicodeScript(input)
        return op.asOutput()
    }

    fun unicodeTranscode(input: Operand<TString>, inputEncoding: String, outputEncoding: String, vararg options: UnicodeTranscode.Options): Output<TString> {
        val op = stringsOps.unicodeTranscode(input, inputEncoding, outputEncoding, *options)
        return op.asOutput()
    }

    fun <T : TNumber, U : TNumber> unsortedSegmentJoin(inputs: Operand<TString>, segmentIds: Operand<T>, numSegments: Operand<U>, vararg options: UnsortedSegmentJoin.Options): Output<TString> {
        val op = stringsOps.unsortedSegmentJoin(inputs, segmentIds, numSegments, *options)
        return op.asOutput()
    }

    fun upper(input: Operand<TString>, vararg options: Upper.Options): Output<TString> {
        val op = stringsOps.upper(input, *options)
        return op.asOutput()
    }
}