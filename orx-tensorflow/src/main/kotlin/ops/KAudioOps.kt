package org.openrndr.extra.tensorflow.ops

import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.AudioOps
import org.tensorflow.op.audio.AudioSpectrogram
import org.tensorflow.op.audio.Mfcc
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32
import org.tensorflow.types.TString

interface KAudioOps {
    val audioOps: AudioOps

    fun audioSpectrogram(input : Operand<TFloat32>, windowSize : Long, stride : Long, options : AudioSpectrogram.Options) : Output<TFloat32> {
        val op = audioOps.audioSpectrogram(input, windowSize, stride, options)
        return op.asOutput()
    }

    fun encodeWav(audio : Operand<TFloat32>, sampleRate : Operand<TInt32>) : Output<TString> {
        val op = audioOps.encodeWav(audio, sampleRate)
        return op.asOutput()
    }

    fun mfcc(spectrogram : Operand<TFloat32>, sampleRate : Operand<TInt32>, options : Mfcc.Options) : Output<TFloat32> {
        val op = audioOps.mfcc(spectrogram, sampleRate, options)
        return op.asOutput()
    }
}