import org.openrndr.application
import org.openrndr.color.ColorRGBa
import org.openrndr.draw.BufferMultisample
import org.openrndr.extensions.SingleScreenshot
import org.openrndr.extra.dnk3.*
import org.openrndr.extra.dnk3.gltf.buildSceneNodes
import org.openrndr.extra.dnk3.gltf.loadGltfFromFile
import org.openrndr.extra.dnk3.query.findNodeByName
import org.openrndr.extra.dnk3.renderers.segmentContourRenderer
import org.openrndr.extras.camera.Orbital
import org.openrndr.math.Vector3
import org.openrndr.math.mod_
import java.io.File

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

        val gltf = loadGltfFromFile(File("demo-data/gltf-models/gundam-01b.glb"))
        val scene = Scene(SceneNode())



        val sceneData = gltf.buildSceneNodes()
        scene.root.children.addAll(sceneData.scenes.first())

        val meshNode = scene.findNodeByName("SketchUp")!!

        val parts = meshNode.findContent { this as? Mesh }
        for (part in parts) {
            println(part)
        }
        println(parts.size)


        val grouped = parts.groupBy { it.node }
        println(grouped.size)

        for (child in meshNode.children) {
            println(child.name)
            println(child.entities.size)
            println(child.children.size)
            println("---")
        }



        // -- create a renderer, try it with BufferMultisample.SampleCount(8) for better results
        val renderer = segmentContourRenderer(BufferMultisample.Disabled)
        extend(Orbital()) {
            far = 500.0
            lookAt = Vector3(0.0, 40.0, 0.0)
            eye = Vector3(150.0, 40.0, 200.0)
            fov = 40.0
        }

        extend {
            //sceneData.animations[2].applyToTargets(seconds.mod_(sceneData.animations[2].duration))
            drawer.clear(ColorRGBa.PINK)
            renderer.draw(drawer, scene)
        }
    }
}