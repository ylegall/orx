package org.openrndr.extra.tensorflow.ops

import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.SummaryOps
import org.tensorflow.op.summary.AudioSummary
import org.tensorflow.op.summary.ImageSummary
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TString
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

interface KSummaryOps {
    val summaryOps: SummaryOps

    fun audioSummary(tag : Operand<TString>, tensor : Operand<TFloat32>, sampleRate : Operand<TFloat32>, options : AudioSummary.Options) : Output<TString> {
        val op = summaryOps.audioSummary(tag, tensor, sampleRate, options)
        return op.asOutput()
    }

    fun <T : TNumber> histogramSummary(tag : Operand<TString>, values : Operand<T>) : Output<TString> {
        val op = summaryOps.histogramSummary(tag, values)
        return op.asOutput()
    }

    fun <T : TNumber> imageSummary(tag : Operand<TString>, tensor : Operand<T>, options : ImageSummary.Options) : Output<TString> {
        val op = summaryOps.imageSummary(tag, tensor, options)
        return op.asOutput()
    }

    fun mergeSummary(inputs : Iterable<Operand<TString>>) : Output<TString> {
        val op = summaryOps.mergeSummary(inputs)
        return op.asOutput()
    }

    fun <T : TNumber> scalarSummary(tags : Operand<TString>, values : Operand<T>) : Output<TString> {
        val op = summaryOps.scalarSummary(tags, values)
        return op.asOutput()
    }

    fun <T : TType> tensorSummary(tag : Operand<TString>, tensor : Operand<T>, serializedSummaryMetadata : Operand<TString>) : Output<TString> {
        val op = summaryOps.tensorSummary(tag, tensor, serializedSummaryMetadata)
        return op.asOutput()
    }
}