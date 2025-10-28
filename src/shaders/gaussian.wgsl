struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) size: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) conic: vec4<f32>,
    @location(3) center: vec2<f32>
};
struct Splat {
    //TODO: information defined in preprocess compute shader
    pos_size: array<u32, 2>,
    conic: array<u32, 2>,
    color_sh: array<u32, 2>,
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<storage, read> sort_indices: array<u32>;
@group(0) @binding(2) var<uniform> camera: CameraUniforms;

@vertex
fn vs_main(
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass
    let idx = sort_indices[instance_idx];
    let splat = splats[idx];

    let xy = unpack2x16float(splat.pos_size[0]);
    let wh = unpack2x16float(splat.pos_size[1]);

    let x = xy.x;
    let y = xy.y;
    let w = wh.x * 2.0;
    let h = wh.y * 2.0;

    let quads = array<vec2f, 6> (
        vec2f(x - w, y + h),
        vec2f(x - w, y - h),
        vec2f(x + w, y - h),
        vec2f(x + w, y - h),
        vec2f(x + w, y + h),
        vec2f(x - w, y + h),
    );

    let conic01 = unpack2x16float(splat.conic[0]);
    let conic23 = unpack2x16float(splat.conic[1]);
    let conic = vec3<f32>(conic01.x, conic01.y, conic23.x);
    let opacity = conic23.y;

    var vertex_out: VertexOutput;

    vertex_out.conic = vec4f(conic, opacity);

    vertex_out.center = vec2f(x, y);

    vertex_out.position = vec4f(quads[vertex_idx].x, quads[vertex_idx].y, 0.0f, 1.0f);

    vertex_out.size = (wh * 0.5f + 0.5f) * camera.viewport.xy;

    let rg = unpack2x16float(splat.color_sh[0]);
    let ba = unpack2x16float(splat.color_sh[1]);
    vertex_out.color = vec3<f32>(rg.x, rg.y, ba.x);

    return vertex_out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var pos = (in.position.xy / camera.viewport) * 2.0f - 1.0f;
    pos.y = -pos.y;

    var offset = pos.xy - in.center.xy;
    offset = vec2f(-offset.x, offset.y) * camera.viewport * 0.5f;

    var power = (in.conic.x * pow(offset.x, 2.0f) + 
                in.conic.z * pow(offset.y, 2.0f))  * 
                -0.5f - 
                in.conic.y * offset.x * offset.y;

    if (power > 0.0f) {
        return vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    }

    let alpha = clamp(in.conic.w * exp(power), 0.0f, 0.99f);

    return vec4<f32>(in.color * alpha, alpha);
}