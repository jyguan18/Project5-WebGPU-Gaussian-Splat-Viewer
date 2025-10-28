const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    pos_size: array<u32, 2>,
    conic: array<u32, 2>,
    color_sh: array<u32, 2>,
};

//TODO: bind your data here
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read> sh_coeffs: array<u32>;
@group(3) @binding(0)
var<storage, read_write> splats: array<Splat>;
@group(3) @binding(1)
var<uniform> render_settings: RenderSettings;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored

    let max_num_coefs = 16u;
    let coef_offset = c_idx * 3u / 2u;
    let base_idx = splat_idx * max_num_coefs * 3u / 2u;
    
    let packed_rg = sh_coeffs[base_idx + coef_offset];
    let packed_b = sh_coeffs[base_idx + coef_offset + 1u];
    let color01 = unpack2x16float(packed_rg);
    let color23 = unpack2x16float(packed_b);

    if (c_idx % 2u == 0u) {
        return vec3<f32>(color01.x, color01.y, color23.x);
    } else {
        return vec3<f32>(color01.y, color23.x, color23.y);
    }    
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction
    if (idx >= arrayLength(&gaussians)){
        return;
    }

    let gaussian = gaussians[idx];

    let pos_packed = gaussian.pos_opacity[0];
    let pos_x = unpack2x16float(pos_packed).x;
    let pos_y = unpack2x16float(pos_packed).y;

    let z_op_packed = gaussian.pos_opacity[1];
    let op_z = unpack2x16float(z_op_packed).x;
    let opacity = unpack2x16float(z_op_packed).y;

    let pos_world = vec4<f32>(pos_x,pos_y,op_z, 1.0);

    // transform to view space
    let pos_view = camera.view * vec4<f32>(pos_world);

    // project to clip
    let pos_clip = camera.proj * pos_view;

    // convert to ndc
    let pos_ndc = pos_clip.xyz / pos_clip.w;

    // convert to screen space
    let pos_screen = vec2<f32>((pos_ndc.x * 0.5 + 0.5) * camera.viewport.x,
    (1.0 - (pos_ndc.y * 0.5 + 0.5) * camera.viewport.y));

    if (pos_ndc.x < -1.2f || pos_ndc.x > 1.2f ||
        pos_ndc.y < -1.2f || pos_ndc.y > 1.2f ||
        (camera.view * pos_world).z < 0.0f )
    {
        return;
    }

    let r01 = unpack2x16float(gaussian.rot[0]);
    let r23 = unpack2x16float(gaussian.rot[1]);
    let rot = vec4<f32>(r01.x, r01.y, r23.x, r23.y);

    let r = rot.x;
    let x = rot.y;
    let y = rot.z;
    let z = rot.w;

    let R = mat3x3f(
        1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - r * z), 2.0f * (x * z + r * y),
        2.0f * (x * y + r * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - r * x),
        2.0f * (x * z - r * y), 2.0f * (y * z + r * x), 1.0f - 2.0f * (x * x + y * y)
    );

    let scale01 = unpack2x16float(gaussian.scale[0]);
    let scale23 = unpack2x16float(gaussian.scale[1]);

    let scale = exp(vec3f(scale01.x, scale01.y, scale23.x));

    let S = mat3x3f(
        scale.x * render_settings.gaussian_scaling, 0.0f, 0.0f,
        0.0f, scale.y * render_settings.gaussian_scaling, 0.0f,
        0.0f, 0.0f, scale.z * render_settings.gaussian_scaling
    );

    let covar_matrix_3D = transpose(S * R) * S * R;

    let covar_3D = array<f32, 6>(
        covar_matrix_3D[0][0],
        covar_matrix_3D[0][1],
        covar_matrix_3D[0][2],
        covar_matrix_3D[1][1],
        covar_matrix_3D[1][2],
        covar_matrix_3D[2][2],
    );

    let J = mat3x3f(
        camera.focal.x / pos_view.z, 0.0f, -(camera.focal.x * pos_view.x) / (pos_view.z * pos_view.z),
        0.0f, camera.focal.y / pos_view.z, -(camera.focal.y * pos_view.y) / (pos_view.z * pos_view.z),
        0.0f, 0.0f, 0.0f
    );

    let W = transpose(mat3x3f(
        camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz
    ));

    let T = W * J;

    let V = mat3x3f(
        covar_3D[0], covar_3D[1], covar_3D[2],
        covar_3D[1], covar_3D[3], covar_3D[4],
        covar_3D[2], covar_3D[4], covar_3D[5],
    );

    var covar_matrix_2D = transpose(T) * transpose(V) * T;
    covar_matrix_2D[0][0] += 0.3f;
    covar_matrix_2D[1][1] += 0.3f;

    let covar_2D = vec3(
        covar_matrix_2D[0][0],
        covar_matrix_2D[0][1],
        covar_matrix_2D[1][1]
    );

    let determinant = covar_2D.x * covar_2D.z - (covar_2D.y * covar_2D.y);

    if (determinant == 0.0f) {
        return;
    }

    let mid = (covar_2D.x + covar_2D.z) * 0.5f;
    let lambda1 = mid + sqrt(max(0.1f, mid * mid - determinant));
    let lambda2 = mid - sqrt(max(0.1f, mid * mid - determinant));
    let radius = ceil(3.0f * sqrt(max(lambda1, lambda2)));


    // depth for sort
    let depth = pos_view.z;
    sort_depths[idx] = bitcast<u32>(100.0f - depth);
    sort_indices[idx] = idx;

    let view_dir = normalize(pos_world.xyz - camera.view_inv[3].xyz);
    let color = computeColorFromSH(view_dir, idx, u32(render_settings.sh_deg));

    splats[idx].pos_size[0] = pack2x16float(pos_ndc.xy);
    splats[idx].pos_size[1] = pack2x16float(vec2(radius, radius) / camera.viewport);

    let conic = vec3f( covar_2D.z / determinant, -covar_2D.y / determinant, covar_2D.x / determinant);
    let conic01 = pack2x16float(conic.xy);
    let conic23 = pack2x16float(vec2(conic.z, 1.0f / (1.0f + exp(-opacity))));
    splats[idx].conic[0] = conic01;
    splats[idx].conic[1] = conic23;
    splats[idx].color_sh[0] = pack2x16float(vec2<f32>(color.r, color.g));
    splats[idx].color_sh[1] = pack2x16float(vec2<f32>(color.b, 1.0f));

    atomicAdd(&sort_infos.keys_size, 1u);

    let keys_per_dispatch = workgroupSize * sortKeyPerThread;

    if (idx % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }

    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys

    let view = camera.view;
    let pos_opacity = gaussians[idx].pos_opacity;
    let passes = sort_infos.passes;
    let sort_depth = sort_depths[0];
    let sort_index = sort_indices[0];
    let dispatch_z = sort_dispatch.dispatch_z;
}