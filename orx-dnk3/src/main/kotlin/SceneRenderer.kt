package org.openrndr.extra.dnk3

import org.openrndr.color.ColorRGBa
import org.openrndr.draw.*
import org.openrndr.extra.dnk3.cubemap.CubemapPassthrough
import org.openrndr.extra.dnk3.cubemap.IrradianceConvolution
import org.openrndr.extra.dnk3.cubemap.evaluateSHIrradiance
import org.openrndr.extra.dnk3.cubemap.irradianceCoefficients
import org.openrndr.extra.fx.blur.ApproximateGaussianBlur
import org.openrndr.math.Matrix44
import org.openrndr.math.Vector3
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class SceneRenderer {

    class Configuration {
        var multisampleLines = false
    }

    val configuration = Configuration()

    val irradianceConvolution = IrradianceConvolution()
    val passthrough = CubemapPassthrough();

    val blur = ApproximateGaussianBlur()

    var shadowLightTargets = mutableMapOf<ShadowLight, RenderTarget>()
    var meshCubemaps = mutableMapOf<Mesh, Cubemap>()

    var cubemapDepthBuffer = depthBuffer(256, 256, DepthFormat.DEPTH16, BufferMultisample.Disabled)


    val tempCubemap = cubemap(256, format = ColorFormat.RGB, type = ColorType.FLOAT32)
    val filteredCubemap = cubemap(256)


    var irradianceArrayCubemap: ArrayCubemap? = null
    var irradianceSHMap: BufferTexture? = null


    var outputPasses = mutableListOf(DefaultOpaquePass, DefaultTransparentPass)
    var outputPassTarget: RenderTarget? = null
    var outputPassTargetMS: RenderTarget? = null

    val postSteps = mutableListOf<PostStep>()
    val buffers = mutableMapOf<String, ColorBuffer>()

    var drawFinalBuffer = true

    var first = true
    fun draw(drawer: Drawer, scene: Scene) {
        drawer.pushStyle()
        drawer.depthWrite = true
        drawer.depthTestPass = DepthTestPass.LESS_OR_EQUAL

        drawer.cullTestPass = CullTestPass.FRONT

        scene.dispatcher.execute()

        // update all the transforms
        scene.root.scan(Matrix44.IDENTITY) { p ->
            if (p !== Matrix44.IDENTITY) {
                worldTransform = p * transform
            } else {
                worldTransform = transform
            }
            worldTransform
        }

        val lights = scene.root.findContent { this as? Light }
        val meshes = scene.root.findContent { this as? Mesh }
        val skinnedMeshes = scene.root.findContent { this as? SkinnedMesh }

        val fogs = scene.root.findContent { this as? Fog }
        val instancedMeshes = scene.root.findContent { this as? InstancedMesh }

        val irradianceProbes = scene.root.findContent { this as? IrradianceProbe }
        val irradianceProbePositions = irradianceProbes.map { it.node.worldPosition }

        run {
            lights.filter { it.content is ShadowLight && (it.content as ShadowLight).shadows is Shadows.MappedShadows }.forEach {
                val shadowLight = it.content as ShadowLight
                val pass: RenderPass
                pass = when (shadowLight.shadows) {
                    is Shadows.PCF, is Shadows.Simple -> {
                        LightPass
                    }
                    is Shadows.VSM -> {
                        VSMLightPass
                    }
                    else -> TODO()
                }
                val target = shadowLightTargets.getOrPut(shadowLight) {
                    val mapSize = (shadowLight.shadows as Shadows.MappedShadows).mapSize
                    pass.createPassTarget(mapSize, mapSize, DepthFormat.DEPTH16)
                }
                target.clearDepth(depth = 1.0)

                val look = shadowLight.view(it.node)
                val materialContext = MaterialContext(pass, lights, fogs, shadowLightTargets, emptyMap(), 0)
                drawer.isolatedWithTarget(target) {
                    drawer.projection = shadowLight.projection(target)
                    drawer.view = look
                    drawer.model = Matrix44.IDENTITY

                    drawer.clear(ColorRGBa.BLACK)
                    drawer.cullTestPass = CullTestPass.FRONT
                    drawPass(drawer, pass, materialContext, meshes, instancedMeshes, skinnedMeshes)
                }
                when (shadowLight.shadows) {
                    is Shadows.VSM -> {
                        blur.gain = 1.0
                        blur.sigma = 3.0
                        blur.window = 9
                        blur.spread = 1.0
                        blur.apply(target.colorBuffer(0), target.colorBuffer(0))
                    }
                }
            }
        }


        run {
            if (irradianceArrayCubemap == null) {
                irradianceArrayCubemap = arrayCubemap(256, 1)
            }
            var probeID = 0

            if (irradianceSHMap == null && irradianceProbes.size > 0) {

                val hash = scene.hashCode()
                if (File("sh-$hash.orb").exists()) {
                    irradianceSHMap = loadBufferTexture(File("sh-$hash.orb"))

                } else {


                    irradianceSHMap = bufferTexture(irradianceProbes.size * 9, format = ColorFormat.RGB, type = ColorType.FLOAT32)
                    val buffer = ByteBuffer.allocateDirect(irradianceProbePositions.size * 9 * 3 * 4)
                    buffer.order(ByteOrder.nativeOrder())

                    for ((node, probe) in irradianceProbes) {
                        if (probe.dirty) {
                            println("rendering probe")
                            val pass = IrradianceProbePass
                            val materialContext = MaterialContext(pass, lights, fogs, shadowLightTargets, emptyMap(), 0)
                            val position = node.worldPosition

                            for (side in CubemapSide.values()) {
                                val target = renderTarget(256, 256) {
                                    this.arrayCubemap(irradianceArrayCubemap!!, side, 0)
                                    this.depthBuffer(cubemapDepthBuffer)
                                }
                                drawer.isolatedWithTarget(target) {
                                    drawer.clear(ColorRGBa.BLACK)
                                    //drawer.perspective(90.0, 1.0, 0.1, 100.0)
                                    drawer.projection = probe.projectionMatrix
                                    drawer.view = Matrix44.IDENTITY
                                    drawer.model = Matrix44.IDENTITY
                                    drawer.lookAt(position, position + side.forward, side.up)
                                    drawPass(drawer, pass, materialContext, meshes, instancedMeshes, skinnedMeshes)
                                }

                                target.detachDepthBuffer()
                                target.detachColorBuffers()
                                target.destroy()
                            }

                            irradianceArrayCubemap!!.copyTo(0, tempCubemap)

                            val coefficients = tempCubemap.irradianceCoefficients()
                            for (coef in coefficients) {
                                buffer.putVector3((coef))
                            }


                            val out = evaluateSHIrradiance(Vector3.Companion.UNIT_Y, coefficients)
                            println(out)

                            println(coefficients.joinToString(", "))
//                    irradianceConvolution.apply(tempCubemap, filteredCubemap)
////                    passthrough.apply(tempCubemap, filteredCubemap)
//                    filteredCubemap.copyTo(irradianceArrayCubemap!!, probeID)


                            probeID++
                            probe.dirty = false
                        }
                    }
                    irradianceSHMap?.let {
                        buffer.rewind()
                        it.write(buffer)

                        it.saveToFile(File("sh-$hash.orb"))
                    }
                }
            }
        }


        run {
            for (pass in outputPasses) {
                val materialContext = MaterialContext(pass, lights, fogs, shadowLightTargets, meshCubemaps, irradianceProbes.size)
                materialContext.irradianceArrayCubemap = irradianceArrayCubemap
                materialContext.irradianceProbePositions = irradianceProbePositions
                materialContext.irradianceSHMap = irradianceSHMap

                val defaultPasses = setOf(DefaultTransparentPass, DefaultOpaquePass)

                if ((pass !in defaultPasses || postSteps.isNotEmpty()) && outputPassTarget == null) {
                    outputPassTarget = pass.createPassTarget(RenderTarget.active.width, RenderTarget.active.height)
                }

                if (pass == outputPasses[0]) {
                    outputPassTarget?.let {
                        drawer.withTarget(it) {
                            clear(ColorRGBa.TRANSPARENT)
                        }
                    }
                }
                outputPassTarget?.let { target ->
                    pass.combiners.forEach {
                        if (it is ColorBufferFacetCombiner) {
                            val index = target.colorBufferIndex(it.targetOutput)
                            target.blendMode(index, it.blendMode)
                        }
                    }
                }
                outputPassTarget?.bind()
                drawPass(drawer, pass, materialContext, meshes, instancedMeshes, skinnedMeshes)
                outputPassTarget?.unbind()

                outputPassTarget?.let { output ->
                    for (combiner in pass.combiners) {
                        buffers[combiner.targetOutput] = output.colorBuffer(combiner.targetOutput)
                    }
                }
            }
            val lightContext = LightContext(lights, shadowLightTargets)
            val postContext = PostContext(lightContext, drawer.view.inversed)

            for (postStep in postSteps) {
                postStep.apply(buffers, postContext)
            }
        }

        drawer.popStyle()
        if (drawFinalBuffer) {
            outputPassTarget?.let { output ->
                drawer.isolated {
                    drawer.ortho()
                    drawer.view = Matrix44.IDENTITY
                    drawer.model = Matrix44.IDENTITY
                    val outputName = (postSteps.last() as FilterPostStep).output
                    val outputBuffer = buffers[outputName]
                            ?: throw IllegalArgumentException("can't find $outputName buffer")
                    drawer.image(outputBuffer)
                }
            }
        }
    }

    private fun drawPass(drawer: Drawer, pass: RenderPass, materialContext: MaterialContext,
                         meshes: List<NodeContent<Mesh>>,
                         instancedMeshes: List<NodeContent<InstancedMesh>>,
                         skinnedMeshes: List<NodeContent<SkinnedMesh>>
    ) {

        drawer.depthWrite = pass.depthWrite
        val primitives = meshes.flatMap { mesh ->
            mesh.content.primitives.map { primitive ->
                NodeContent(mesh.node, primitive)
            }
        }

        // -- draw all meshes
        primitives
                .filter { (it.content.material.transparent && pass.renderTransparent) || (!it.content.material.transparent && pass.renderOpaque) }
                .forEach {
                    val primitive = it.content
                    drawer.isolated {
                        if (primitive.material.doubleSided) {
                            drawer.drawStyle.cullTestPass = CullTestPass.ALWAYS
                        }
                        val hasNormalAttribute = primitive.geometry.vertexBuffers.any { it.vertexFormat.hasAttribute("normal") }
                        val primitiveContext = PrimitiveContext(hasNormalAttribute, false)
                        val shadeStyle = primitive.material.generateShadeStyle(materialContext, primitiveContext)
                        shadeStyle.parameter("viewMatrixInverse", drawer.view.inversed)
                        primitive.material.applyToShadeStyle(materialContext, shadeStyle)
                        drawer.shadeStyle = shadeStyle
                        drawer.model = it.node.worldTransform

                        if (primitive.geometry.indexBuffer == null) {
                            drawer.vertexBuffer(primitive.geometry.vertexBuffers,
                                    primitive.geometry.primitive,
                                    primitive.geometry.offset,
                                    primitive.geometry.vertexCount)
                        } else {
                            drawer.vertexBuffer(primitive.geometry.indexBuffer!!,
                                    primitive.geometry.vertexBuffers,
                                    primitive.geometry.primitive,
                                    primitive.geometry.offset,
                                    primitive.geometry.vertexCount)
                        }
                    }
                }


        val skinnedPrimitives = skinnedMeshes.flatMap { mesh ->
            mesh.content.primitives.map { primitive ->
                NodeContent(mesh.node, Pair(primitive, mesh))
            }
        }

        skinnedPrimitives
                .filter {
                    (it.content.first.material.transparent && pass.renderTransparent) ||
                            (!it.content.first.material.transparent && pass.renderOpaque)
                }
                .forEach {
                    val primitive = it.content.first
                    val skinnedMesh = it.content.second.content
                    drawer.isolated {
                        if (primitive.material.doubleSided) {
                            drawer.drawStyle.cullTestPass = CullTestPass.ALWAYS
                        }
                        val hasNormalAttribute = primitive.geometry.vertexBuffers.any { it.vertexFormat.hasAttribute("normal") }
                        val primitiveContext = PrimitiveContext(hasNormalAttribute, true)

                        val nodeInverse = it.node.worldTransform.inversed


                        val jointTransforms = (skinnedMesh.joints zip skinnedMesh.inverseBindMatrices)
                                .map { (nodeInverse * it.first.worldTransform * it.second) }
//                        val jointNormalTransforms = jointTransforms.map { Matrix44.IDENTITY }

                        val shadeStyle = primitive.material.generateShadeStyle(materialContext, primitiveContext)

                        shadeStyle.parameter("jointTransforms", jointTransforms.toTypedArray())
//                        shadeStyle.parameter("jointNormalTransforms", jointNormalTransforms.toTypedArray())

                        shadeStyle.parameter("viewMatrixInverse", drawer.view.inversed)
                        primitive.material.applyToShadeStyle(materialContext, shadeStyle)
                        drawer.shadeStyle = shadeStyle
                        drawer.model = it.node.worldTransform

                        if (primitive.geometry.indexBuffer == null) {
                            drawer.vertexBuffer(primitive.geometry.vertexBuffers,
                                    primitive.geometry.primitive,
                                    primitive.geometry.offset,
                                    primitive.geometry.vertexCount)
                        } else {
                            drawer.vertexBuffer(primitive.geometry.indexBuffer!!,
                                    primitive.geometry.vertexBuffers,
                                    primitive.geometry.primitive,
                                    primitive.geometry.offset,
                                    primitive.geometry.vertexCount)
                        }
                    }
                }


        val instancedPrimitives = instancedMeshes.flatMap { mesh ->
            mesh.content.primitives.map { primitive ->
                NodeContent(mesh.node, MeshPrimitiveInstance(primitive, mesh.content.instances, mesh.content.attributes))
            }
        }

        // -- draw all instanced meshes
        instancedPrimitives
                .filter { (it.content.primitive.material.transparent && pass.renderTransparent) || (!it.content.primitive.material.transparent && pass.renderOpaque) }
                .forEach {
                    val primitive = it.content
                    drawer.isolated {
                        val primitiveContext = PrimitiveContext(true, false)
                        val shadeStyle = primitive.primitive.material.generateShadeStyle(materialContext, primitiveContext)
                        shadeStyle.parameter("viewMatrixInverse", drawer.view.inversed)
                        primitive.primitive.material.applyToShadeStyle(materialContext, shadeStyle)
                        if (primitive.primitive.material.doubleSided) {
                            drawer.drawStyle.cullTestPass = CullTestPass.ALWAYS
                        }
                        drawer.shadeStyle = shadeStyle
                        drawer.model = it.node.worldTransform
                        drawer.vertexBufferInstances(primitive.primitive.geometry.vertexBuffers,
                                primitive.attributes,
                                DrawPrimitive.TRIANGLES,
                                primitive.instances,
                                primitive.primitive.geometry.offset,
                                primitive.primitive.geometry.vertexCount)
                    }
                }
        drawer.depthWrite = true
    }
}

fun sceneRenderer(builder: SceneRenderer.() -> Unit): SceneRenderer {
    val sceneRenderer = SceneRenderer()
    sceneRenderer.builder()
    return sceneRenderer
}

private fun ByteBuffer.putVector3(v: Vector3) {
    putFloat(v.x.toFloat())
    putFloat(v.y.toFloat())
    putFloat(v.z.toFloat())
}