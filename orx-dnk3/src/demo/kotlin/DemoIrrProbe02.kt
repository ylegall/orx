import kotlinx.coroutines.yield
import org.openrndr.KEY_ARROW_LEFT
import org.openrndr.KEY_ARROW_RIGHT
import org.openrndr.application
import org.openrndr.color.ColorRGBa
import org.openrndr.draw.*
import org.openrndr.extensions.SingleScreenshot
import org.openrndr.extra.dnk3.*
import org.openrndr.extra.dnk3.features.IrradianceSH
import org.openrndr.extra.dnk3.features.addIrradianceSH
import org.openrndr.extra.dnk3.gltf.buildSceneNodes
import org.openrndr.extra.dnk3.gltf.loadGltfFromFile
import org.openrndr.extra.dnk3.post.VolumetricIrradiance
import org.openrndr.extra.dnk3.query.allMaterials
import org.openrndr.extra.dnk3.query.findMaterialByName
import org.openrndr.extra.dnk3.renderers.dryRenderer
import org.openrndr.extra.dnk3.renderers.postRenderer
import org.openrndr.extra.dnk3.tools.addSkybox
import org.openrndr.extra.fx.tonemap.Uncharted2Tonemap
import org.openrndr.extras.camera.Orbital
import org.openrndr.extras.meshgenerators.boxMesh
import org.openrndr.extras.meshgenerators.sphereMesh
import org.openrndr.ffmpeg.ScreenRecorder
import org.openrndr.filter.color.Delinearize
import org.openrndr.launch
import org.openrndr.math.*
import org.openrndr.math.transforms.transform
import java.io.File
import kotlin.math.cos
import kotlin.math.sin

fun main() = application {
    configure {
        width = 1280
        height = 720
        //multisample = WindowMultisample.SampleCount(8)
    }

    program {
        if (System.getProperty("takeScreenshot") == "true") {
            extend(SingleScreenshot()) {
                this.outputFile = System.getProperty("screenshotPath")
            }
        }

        val gltf = loadGltfFromFile(File("d:\\blender\\street-01.glb"))
        val scene = Scene(SceneNode())

        val c = 10
        scene.addSkybox("file:d:/hdr-probes-2/moonless-golf/output_pmrem.dds", intensity = 0.125 * 0.5)
        scene.addIrradianceSH(c, c, c, 40.0/c, cubemapSize = 32, offset = Vector3(0.0, 0.0, 0.0))

        val sceneData = gltf.buildSceneNodes()
        scene.root.children.addAll(sceneData.scenes.first())

        val cubemap = Cubemap.fromUrl("file:d:/hdr-probes-2/moonless-golf/output_pmrem.dds")
        cubemap.filter(MinifyingFilter.LINEAR_MIPMAP_LINEAR, MagnifyingFilter.LINEAR)
        for (material in scene.allMaterials()) {
            if (material is PBRMaterial) {
                material.cubemapReflection = CubemapReflection(cubemap).apply { color = ColorRGBa.WHITE.shade(0.125 * 0.5)}
            }
        }

        // -- create a renderer
        val renderer = postRenderer()

        // -- setup up volumetric fog
        renderer.postSteps.add(
                FilterPostStep(1.0, VolumetricIrradiance(), listOf("color", "clipDepth"), "volumetric-irradiance", ColorFormat.RGB, ColorType.FLOAT16) {
                    this.stepLength = 0.1
                    this.irradianceSH = scene.features[0] as IrradianceSH
                    this.projectionMatrixInverse = drawer.projection.inversed
                    this.viewMatrixInverse = drawer.view.inversed
                }
        )

        // -- set up tone mapping
        renderer.postSteps.add(
                FilterPostStep(1.0, Uncharted2Tonemap(), listOf("volumetric-irradiance"), "ldr", ColorFormat.RGB, ColorType.FLOAT16)
        )

        val orb = extend(Orbital()) {
            camera.setView(Vector3(-0.49, -0.24, 0.20), Spherical(26.56, 90.0, 6.533), 40.0)
        }

        // -- draw here for preproc
        renderer.draw(drawer, scene)

        extend {
            drawer.clear(ColorRGBa.BLACK)
            renderer.draw(drawer, scene)
            drawer.defaults()
        }
    }
}