import org.tensorflow.EagerSession
import org.tensorflow.op.Ops

fun <T> eager(f: GraphBuilder.() -> T): T {
    val session = EagerSession.options().async(false).build()
    val tf = Ops.create(session)
    val scope = tf.scope()
    println("running graph")
    val gb = GraphBuilder(scope)
    val retval = gb.f()
    println("done and closing")
    session.close()
    println("returning")
    return retval
}

