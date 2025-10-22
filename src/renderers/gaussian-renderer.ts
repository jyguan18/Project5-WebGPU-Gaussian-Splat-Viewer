import { PointCloud } from "../utils/load";
import preprocessWGSL from "../shaders/preprocess.wgsl";
import renderWGSL from "../shaders/gaussian.wgsl";
import { get_sorter, c_histogram_block_rows, C } from "../sort/sort";
import { Renderer } from "./renderer";

export interface GaussianRenderer extends Renderer {}

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
  const bytesPerSplat = 48;
  const splatBufferSize = pc.num_points * bytesPerSplat;
  const splat_buffer = createBuffer(
    device,
    "splat buffer",
    splatBufferSize,
    GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
  );

  const indirect_buffer_data = new Uint32Array([6, pc.num_points, 0, 0]);

  const indirect_buffer = createBuffer(
    device,
    "indirect buffer",
    indirect_buffer_data.byteLength,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    indirect_buffer_data
  );

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  // const preprocess_pipeline = device.createComputePipeline({
  //   label: "preprocess",
  //   layout: "auto",
  //   compute: {
  //     module: device.createShaderModule({ code: preprocessWGSL }),
  //     entryPoint: "preprocess",
  //     constants: {
  //       workgroupSize: C.histogram_wg_size,
  //       sortKeyPerThread: c_histogram_block_rows,
  //     },
  //   },
  // });

  // const sort_bind_group = device.createBindGroup({
  //   label: "sort",
  //   layout: preprocess_pipeline.getBindGroupLayout(2),
  //   entries: [
  //     { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
  //     {
  //       binding: 1,
  //       resource: { buffer: sorter.ping_pong[0].sort_depths_buffer },
  //     },
  //     {
  //       binding: 2,
  //       resource: { buffer: sorter.ping_pong[0].sort_indices_buffer },
  //     },
  //     {
  //       binding: 3,
  //       resource: { buffer: sorter.sort_dispatch_indirect_buffer },
  //     },
  //   ],
  // });

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const render = device.createRenderPipeline({
    label: "render",
    layout: "auto",
    vertex: {
      module: device.createShaderModule({
        code: renderWGSL,
      }),
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 8,
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
        },
      ],
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

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const quad_verts = new Float32Array([
    -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5,
  ]);

  const quad_vertex_buffer = createBuffer(
    device,
    "quad verts",
    quad_verts.byteLength,
    GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    quad_verts
  );

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      sorter.sort(encoder);

      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: texture_view,
            loadOp: "clear",
            clearValue: [0, 0, 0, 1],
            storeOp: "store",
          },
        ],
      });

      pass.setPipeline(render);
      pass.setVertexBuffer(0, quad_vertex_buffer);
      pass.drawIndirect(indirect_buffer, 0);
      pass.end();
    },
    camera_buffer,
  };
}
