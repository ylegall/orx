#version 330 core

#pragma import org.openrndr.extra.shaderphrases.phrases.Depth.projectionToViewCoordinate;
#pragma import org.openrndr.extra.dnk3.cubemap.SphericalHarmonics.glslEvaluateSH;
#pragma import org.openrndr.extra.dnk3.cubemap.SphericalHarmonics.glslFetchSH;

in vec2 v_texCoord0;
uniform sampler2D tex0; // image
uniform sampler2D tex1; // projDepth

uniform samplerBuffer shMap;
uniform ivec3 shMapDimensions;
uniform vec3 shMapOffset;
uniform float shMapSize;

uniform mat4 projectionMatrixInverse;
uniform mat4 viewMatrixInverse;

out vec4 o_output;

void main() {
    float projDepth = texture(tex0, v_texCoord0).r;
    vec3 viewCoordinate = projectionToViewCoordinate(v_texCoord0, projDepth, projectionMatrixInverse);

    vec3 worldCoordinate = (viewMatrixInverse * vec4(viewCoordinate, 1.0)).xyz;
    vec3 cameraPosition = (viewMatrixInverse * vec4(vec3(0.0), 1.0)).xyz;

    // trace in world space
    vec3 traverse = worldCoordinate - cameraPosition;
    vec3 direction = normalize(traverse);
    int steps = length(traverse);
    vec3 step = traverse / steps;

    vec3 marchPosition = cameraPosition;
    for (int stepIndex = 0; stepIndex < steps; ++stepIndex) {
        position += step;
    }

}