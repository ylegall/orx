package org.openrndr.extra.dnk3.materials

import org.openrndr.draw.ShadeStyle
import org.openrndr.draw.shadeStyle
import org.openrndr.extra.dnk3.Material
import org.openrndr.extra.dnk3.MaterialContext
import org.openrndr.extra.dnk3.PrimitiveContext

class IrradianceDebugMaterial : Material {
    override var doubleSided: Boolean = false

    override var transparent: Boolean = false
    override val fragmentID: Int = 0

    var irradianceProbeID = 0

    override fun generateShadeStyle(context: MaterialContext, primitiveContext: PrimitiveContext): ShadeStyle {
        return shadeStyle {
            fragmentPreamble = """
                vec3 f_emission = vec3(0.0);
            """.trimIndent()

            if (context.irradianceArrayCubemap != null) {
                fragmentTransform = """
                x_fill.rgb = texture(p_irradianceMap, vec4(normalize(va_position), float(p_irradianceProbeID))).rgb;
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
        shadeStyle.parameter("irradianceProbeID", irradianceProbeID)
    }
}