package org.openrndr.extra.dnk3.post

import org.openrndr.draw.*
import org.openrndr.extra.dnk3.features.IrradianceSH
import org.openrndr.extra.shaderphrases.preprocessShader
import org.openrndr.resourceUrl
import java.net.URL

fun preprocessedFilterShaderFromUrl(url: String): Shader {
    return filterShaderFromCode( preprocessShader(URL(url).readText()), "filter-shader: $url")
}

fun preprocessedFilterShaderFromCode(fragmentShaderCode: String, name: String): Shader {
    return Shader.createFromCode(Filter.filterVertexCode, fragmentShaderCode, name)
}

class VolumetricIrradiance : Filter(preprocessedFilterShaderFromUrl(resourceUrl("/shaders/volumetric-irradiance.frag"))) {
    var irradianceSH: IrradianceSH? = null
    override fun apply(source: Array<ColorBuffer>, target: Array<ColorBuffer>) {
        irradianceSH?.shMap?.let {
            parameters["shMap"] = it
        }
        irradianceSH?.let {
            parameters[""]
        }
        super.apply(source, target)
    }
}

