package org.openrndr.extra.tensorflow.ops

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.ImageOps
import org.tensorflow.op.image.*
import org.tensorflow.types.*
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

interface KImageOps {
    val imageOps: ImageOps

    fun <T : TNumber> adjustContrast(images : Operand<T>, contrastFactor : Operand<TFloat32>) : Output<T> {
        val op = imageOps.adjustContrast(images, contrastFactor)
        return op.asOutput()
    }

    fun <T : TNumber> adjustHue(images : Operand<T>, delta : Operand<TFloat32>) : Output<T> {
        val op = imageOps.adjustHue(images, delta)
        return op.asOutput()
    }

    fun <T : TNumber> adjustSaturation(images : Operand<T>, scale : Operand<TFloat32>) : Output<T> {
        val op = imageOps.adjustSaturation(images, scale)
        return op.asOutput()
    }

    fun <T : TNumber> cropAndResize(image : Operand<T>, boxes : Operand<TFloat32>, boxInd : Operand<TInt32>, cropSize : Operand<TInt32>, options : CropAndResize.Options) : Output<TFloat32> {
        val op = imageOps.cropAndResize(image, boxes, boxInd, cropSize, options)
        return op.asOutput()
    }

    fun <T : TNumber> cropAndResizeGradBoxes(grads : Operand<TFloat32>, image : Operand<T>, boxes : Operand<TFloat32>, boxInd : Operand<TInt32>, options : CropAndResizeGradBoxes.Options) : Output<TFloat32> {
        val op = imageOps.cropAndResizeGradBoxes(grads, image, boxes, boxInd, options)
        return op.asOutput()
    }

    fun <T : TNumber> cropAndResizeGradImage(grads : Operand<TFloat32>, boxes : Operand<TFloat32>, boxInd : Operand<TInt32>, imageSize : Operand<TInt32>, t : DataType<T>, options : CropAndResizeGradImage.Options) : Output<T> {
        val op = imageOps.cropAndResizeGradImage(grads, boxes, boxInd, imageSize, t, options)
        return op.asOutput()
    }

    fun decodeAndCropJpeg(contents : Operand<TString>, cropWindow : Operand<TInt32>, options : DecodeAndCropJpeg.Options) : Output<TUint8> {
        val op = imageOps.decodeAndCropJpeg(contents, cropWindow, options)
        return op.asOutput()
    }

    fun decodeBmp(contents : Operand<TString>, options : DecodeBmp.Options) : Output<TUint8> {
        val op = imageOps.decodeBmp(contents, options)
        return op.asOutput()
    }

    fun decodeGif(contents : Operand<TString>) : Output<TUint8> {
        val op = imageOps.decodeGif(contents)
        return op.asOutput()
    }

    fun decodeJpeg(contents : Operand<TString>, options : DecodeJpeg.Options) : Output<TUint8> {
        val op = imageOps.decodeJpeg(contents, options)
        return op.asOutput()
    }

    fun decodePng(contents : Operand<TString>, options : DecodePng.Options) : Output<TUint8> {
        val op = imageOps.decodePng(contents, options)
        return op.asOutput()
    }

    fun <T : TNumber> decodePng(contents : Operand<TString>, dtype : DataType<T>, options : DecodePng.Options) : Output<T> {
        val op = imageOps.decodePng(contents, dtype, options)
        return op.asOutput()
    }

    fun <T : TNumber> drawBoundingBoxes(images : Operand<T>, boxes : Operand<TFloat32>, colors : Operand<TFloat32>) : Output<T> {
        val op = imageOps.drawBoundingBoxes(images, boxes, colors)
        return op.asOutput()
    }

    fun encodeJpeg(image : Operand<TUint8>, options : EncodeJpeg.Options) : Output<TString> {
        val op = imageOps.encodeJpeg(image, options)
        return op.asOutput()
    }

    fun encodeJpegVariableQuality(images : Operand<TUint8>, quality : Operand<TInt32>) : Output<TString> {
        val op = imageOps.encodeJpegVariableQuality(images, quality)
        return op.asOutput()
    }

    fun <T : TNumber> encodePng(image : Operand<T>, options : EncodePng.Options) : Output<TString> {
        val op = imageOps.encodePng(image, options)
        return op.asOutput()
    }

    fun <T : TType> extractImagePatches(images : Operand<T>, ksizes : List<Long>, strides : List<Long>, rates : List<Long>, padding : String) : Output<T> {
        val op = imageOps.extractImagePatches(images, ksizes, strides, rates, padding)
        return op.asOutput()
    }

    fun extractJpegShape(contents : Operand<TString>) : Output<TInt32> {
        val op = imageOps.extractJpegShape(contents)
        return op.asOutput()
    }

    fun <T : TNumber> extractJpegShape(contents : Operand<TString>, outputType : DataType<T>) : Output<T> {
        val op = imageOps.extractJpegShape(contents, outputType)
        return op.asOutput()
    }

    fun <T : TNumber> hsvToRgb(images : Operand<T>) : Output<T> {
        val op = imageOps.hsvToRgb(images)
        return op.asOutput()
    }

    fun nonMaxSuppressionWithOverlaps(overlaps : Operand<TFloat32>, scores : Operand<TFloat32>, maxOutputSize : Operand<TInt32>, overlapThreshold : Operand<TFloat32>, scoreThreshold : Operand<TFloat32>) : Output<TInt32> {
        val op = imageOps.nonMaxSuppressionWithOverlaps(overlaps, scores, maxOutputSize, overlapThreshold, scoreThreshold)
        return op.asOutput()
    }

    fun <T : TNumber> randomCrop(image : Operand<T>, size : Operand<TInt64>, options : RandomCrop.Options) : Output<T> {
        val op = imageOps.randomCrop(image, size, options)
        return op.asOutput()
    }

    fun <T : TNumber> resizeArea(images : Operand<T>, size : Operand<TInt32>, options : ResizeArea.Options) : Output<TFloat32> {
        val op = imageOps.resizeArea(images, size, options)
        return op.asOutput()
    }

    fun <T : TNumber> resizeBicubic(images : Operand<T>, size : Operand<TInt32>, options : ResizeBicubic.Options) : Output<TFloat32> {
        val op = imageOps.resizeBicubic(images, size, options)
        return op.asOutput()
    }

    fun <T : TNumber> resizeBilinear(images : Operand<T>, size : Operand<TInt32>, options : ResizeBilinear.Options) : Output<TFloat32> {
        val op = imageOps.resizeBilinear(images, size, options)
        return op.asOutput()
    }

    fun <T : TNumber> resizeNearestNeighbor(images : Operand<T>, size : Operand<TInt32>, options : ResizeNearestNeighbor.Options) : Output<T> {
        val op = imageOps.resizeNearestNeighbor(images, size, options)
        return op.asOutput()
    }

    fun <T : TNumber> rgbToHsv(images : Operand<T>) : Output<T> {
        val op = imageOps.rgbToHsv(images)
        return op.asOutput()
    }

    fun <T : TNumber> scaleAndTranslate(images : Operand<T>, size : Operand<TInt32>, scale : Operand<TFloat32>, translation : Operand<TFloat32>, options : ScaleAndTranslate.Options) : Output<TFloat32> {
        val op = imageOps.scaleAndTranslate(images, size, scale, translation, options)
        return op.asOutput()
    }
}