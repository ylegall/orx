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

    override fun generateShadeStyle(context: MaterialContext, primitiveContext: PrimitiveContext): ShadeStyle {
        return shadeStyle {
            fragmentPreamble = """
                $glslEvaluateSH
                $glslFetchSH
                ${glslGatherSH(context.irradianceSH!!.xCount, context.irradianceSH!!.yCount, context.irradianceSH!!.zCount, context.irradianceSH!!.spacing, context.irradianceSH!!.offset)}
                vec3 f_emission = vec3(0.0);
            """

            if (context.irradianceSH != null) {
                fragmentTransform = """
                    vec3[9] sh;
                    gatherSH(p_shMap, v_worldPosition, sh);
                x_fill.rgb = evaluateSH(normalize(v_worldNormal), sh);
                
            """.trimIndent()
            } else {
                fragmentTransform = """
                    discard;
                    """
            }
        }
    }

    override fun applyToShadeStyle(context: MaterialContext, shadeStyle: ShadeStyle) {
        context.irradianceSH?.shMap?.let {
            shadeStyle.parameter("shMap", it)
        }

    }
}