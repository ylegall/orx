package org.openrndr.extra.tensorflow

import org.tensorflow.EagerSession
import org.tensorflow.op.Ops

fun <T> eager(f: GraphBuilder.() -> T): T {
    val session = EagerSession.options().async(false).build()
    val tf = Ops.create(session)
    val scope = tf.scope()
    val gb = GraphBuilder(session)
    val retval = gb.f()
    session.close()
    return retval
}

