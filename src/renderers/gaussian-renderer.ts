import { PointCloud } from "../utils/load";
import preprocessWGSL from "../shaders/preprocess.wgsl";
import renderWGSL from "../shaders/gaussian.wgsl";
import { get_sorter, c_histogram_block_rows, C } from "../sort/sort";
import { Renderer } from "./renderer";

export interface GaussianRenderer extends Renderer {
  render_settings_buffer: GPUBuffer;
}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer
): GaussianRenderer {
  const sorter = get_sorter(pc.num_points, device);

  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);
  const nulling_buffer = createBuffer(
    device,
    "null_buffer",
    nulling_data.byteLength,
    GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    nulling_data
  );

  const bytesPerSplat = 24;
  const splatBufferSize = pc.num_points * bytesPerSplat;
  const splat_buffer = createBuffer(
    device,
    "splat buffer",
    splatBufferSize,
    GPUBufferUsage.STORAGE,
    null
  );

  const indirect_buffer_data = new Uint32Array([6, 0, 0, 0]);

  const indirect_buffer = createBuffer(
    device,
    "indirect buffer",
    indirect_buffer_data.byteLength,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    indirect_buffer_data
  );

  const render_settings_buffer = createBuffer(
    device,
    "render settings buffer",
    8,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    new Float32Array([1.0, pc.sh_deg])
  );

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const render_pipeline = device.createRenderPipeline({
    label: "render",
    layout: "auto",
    vertex: {
      module: device.createShaderModule({
        code: renderWGSL,
      }),
      entryPoint: "vs_main",
      buffers: [],
    },
    fragment: {
      module: device.createShaderModule({
        code: renderWGSL,
      }),
      entryPoint: "fs_main",
      targets: [{ format: presentation_format }],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  const render_pipeline_bind_group = device.createBindGroup({
    label: "render pipeline bind group",
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer: splat_buffer },
      },
      {
        binding: 1,
        resource: { buffer: sorter.ping_pong[0].sort_indices_buffer },
      },
    ],
  });

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: "preprocess",
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: "preprocess",
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const preprocess_camera_bind_group = device.createBindGroup({
    label: "preprocess camera",
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer: camera_buffer },
      },
    ],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: "preprocess data",
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      {
        binding: 0,
        resource: { buffer: pc.gaussian_3d_buffer },
      },
      {
        binding: 1,
        resource: { buffer: pc.sh_buffer },
      },
    ],
  });

  const compute_pipeline_bind_group = device.createBindGroup({
    label: "compute pipeline bind group",
    layout: preprocess_pipeline.getBindGroupLayout(3),
    entries: [
      {
        binding: 0,
        resource: { buffer: splat_buffer },
      },
      {
        binding: 1,
        resource: { buffer: render_settings_buffer },
      },
    ],
  });

  const sort_bind_group = device.createBindGroup({
    label: "sort",
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      {
        binding: 1,
        resource: { buffer: sorter.ping_pong[0].sort_depths_buffer },
      },
      {
        binding: 2,
        resource: { buffer: sorter.ping_pong[0].sort_indices_buffer },
      },
      {
        binding: 3,
        resource: { buffer: sorter.sort_dispatch_indirect_buffer },
      },
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ==============================================

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      // reset sorting buffers
      encoder.copyBufferToBuffer(
        nulling_buffer,
        0,
        sorter.sort_info_buffer,
        0,
        4
      );

      encoder.copyBufferToBuffer(
        nulling_buffer,
        0,
        sorter.sort_dispatch_indirect_buffer,
        0,
        4
      );

      // start compute pass
      const preprocess_pass = encoder.beginComputePass({ label: "preprocess" });
      preprocess_pass.setPipeline(preprocess_pipeline);
      preprocess_pass.setBindGroup(0, preprocess_camera_bind_group);
      preprocess_pass.setBindGroup(1, gaussian_bind_group);
      preprocess_pass.setBindGroup(2, sort_bind_group);
      preprocess_pass.setBindGroup(3, compute_pipeline_bind_group);

      const workgroups = Math.ceil(pc.num_points / C.histogram_wg_size);
      preprocess_pass.dispatchWorkgroups(workgroups);
      preprocess_pass.end();

      sorter.sort(encoder);

      encoder.copyBufferToBuffer(
        sorter.sort_info_buffer,
        0,
        indirect_buffer,
        4,
        4
      );

      // start render pass
      const render_pass = encoder.beginRenderPass({
        label: "render pass",
        colorAttachments: [
          {
            view: texture_view,
            loadOp: "clear",
            clearValue: [0, 0, 0, 1],
            storeOp: "store",
          },
        ],
      });

      render_pass.setPipeline(render_pipeline);
      render_pass.setBindGroup(0, render_pipeline_bind_group);
      render_pass.drawIndirect(indirect_buffer, 0);
      render_pass.end();
    },
    camera_buffer,
    render_settings_buffer,
  };
}
