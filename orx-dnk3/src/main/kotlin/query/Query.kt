package org.openrndr.extra.dnk3.query

import org.openrndr.extra.dnk3.Scene
import org.openrndr.extra.dnk3.SceneNode

fun Scene.findNodeByName(name: String): SceneNode? {
    return root.findNodeByName(name)
}

fun SceneNode.findNodeByName(name : String): SceneNode? {

    if (this.name == name) {
        return this
    } else {
        for (child in children) {
            val candidate = child.findNodeByName(name)
            if (candidate != null) {
                return candidate
            }
        }
    }
    return null
}