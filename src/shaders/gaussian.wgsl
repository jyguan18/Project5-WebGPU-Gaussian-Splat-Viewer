struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
};
struct Splat {
    //TODO: information defined in preprocess compute shader
    pos_size: array<u32, 2>,
    conic: array<u32, 2>,
    color_sh: array<u32, 2>,
};

@group(0) @binding(0) var<storage, read> splats: array<Splat>;

@vertex
fn vs_main(
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass
    let idx = instance_idx;
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

    var vertex_out: VertexOutput;

    vertex_out.position = vec4f(quads[vertex_idx].x, quads[vertex_idx].y, 0.0f, 1.0f);
    return vertex_out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1., 0., 0., 1.);
}