package org.openrndr.extra.dnk3.materials

import org.openrndr.draw.ShadeStyle
import org.openrndr.draw.shadeStyle
import org.openrndr.extra.dnk3.Material
import org.openrndr.extra.dnk3.MaterialContext
import org.openrndr.extra.dnk3.PrimitiveContext
import org.openrndr.extra.dnk3.cubemap.glslEvaluateSH
import org.openrndr.extra.dnk3.cubemap.glslFetchSH
import org.openrndr.extra.dnk3.cubemap.glslGatherSH

class IrradianceDebugMaterial : Material {
    override var doubleSided: Boolean = false
    override var transparent: Boolean = false
    override val fragmentID: Int = 0
    var irradianceProbeID = 0

    override fun generateShadeStyle(context: MaterialContext, primitiveContext: PrimitiveContext): ShadeStyle {
        return shadeStyle {
            fragmentPreamble = """
                $glslEvaluateSH
                $glslFetchSH
                ${glslGatherSH(3, 3, 3, 1.0)}
                vec3 f_emission = vec3(0.0);
            """.trimIndent()

            if (context.irradianceArrayCubemap != null) {
                fragmentTransform = """
                    vec3[9] sh;
                    //fetchSH(p_shMap, p_irradianceProbeID, sh);
                    gatherSH(p_shMap, v_worldPosition, sh);
                //x_fill.rgb = texture(p_irradianceMap, vec4(normalize(va_position), float(p_irradianceProbeID))).rgb;
                x_fill.rgb = evaluateSH(v_worldNormal, sh);
                
            """.trimIndent()
            } else {
                fragmentTransform = """
                    discard;
                    """
            }
        }
    }

    override fun applyToShadeStyle(context: MaterialContext, shadeStyle: ShadeStyle) {
        context.irradianceArrayCubemap?.let {
            shadeStyle.parameter("irradianceMap", it)
        }
        context.irradianceSHMap?.let {
            shadeStyle.parameter("shMap", it)
        }
        shadeStyle.parameter("irradianceProbeID", irradianceProbeID)
    }
}