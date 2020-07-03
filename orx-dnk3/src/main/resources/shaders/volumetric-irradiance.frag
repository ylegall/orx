#version 330 core

#pragma import org.openrndr.extra.shaderphrases.phrases.Depth.projectionToViewCoordinate;
#pragma import org.openrndr.extra.dnk3.cubemap.SphericalHarmonicsKt.glslFetchSH0;
#pragma import org.openrndr.extra.dnk3.cubemap.SphericalHarmonicsKt.glslGridCoordinates;
#pragma import org.openrndr.extra.dnk3.cubemap.SphericalHarmonicsKt.glslGridIndex;
#pragma import org.openrndr.extra.dnk3.cubemap.SphericalHarmonicsKt.glslGatherSH0;

in vec2 v_texCoord0;
uniform sampler2D tex0; // image
uniform sampler2D tex1; // projDepth

uniform samplerBuffer shMap;
uniform ivec3 shMapDimensions;
uniform vec3 shMapOffset;
uniform float shMapSpacing;

uniform mat4 projectionMatrixInverse;
uniform mat4 viewMatrixInverse;
uniform float stepLength;

out vec4 o_output;

void main() {
    vec3 inputColor = texture(tex0, v_texCoord0).rgb;
    float projDepth = texture(tex1, v_texCoord0).r;
    vec3 viewCoordinate = projectionToViewCoordinate(v_texCoord0, projDepth, projectionMatrixInverse);

    vec3 worldCoordinate = (viewMatrixInverse * vec4(viewCoordinate, 1.0)).xyz;
    vec3 cameraPosition = (viewMatrixInverse * vec4(vec3(0.0), 1.0)).xyz;

    // trace in world space
    vec3 traverse = worldCoordinate - cameraPosition;
    vec3 direction = normalize(traverse);
    int steps = min(100, int(length(traverse) / stepLength));
    vec3 step = traverse / steps;

    vec3 marchPosition = cameraPosition;
    vec3 accumulated = vec3(0.0);
    for (int stepIndex = 0; stepIndex < steps; ++stepIndex) {

        vec3 sh0;
        gatherSH0(shMap, marchPosition, shMapDimensions, shMapOffset, shMapSpacing, sh0);
        accumulated += sh0*0.1;
        marchPosition += step;
    }

    o_output = vec4(accumulated + inputColor, 1.0);


}
