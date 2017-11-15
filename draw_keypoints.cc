#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"


using namespace tensorflow;

REGISTER_OP("DrawKeypoints")
    .Input("images: float32")
    .Input("keypoints: float32")
    .Output("drawed: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class DrawKeypointsOp : public OpKernel {
 public:
  explicit DrawKeypointsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& images = context->input(0);
    const Tensor& keypoints = context->input(1);
    const int64 depth = images.dim_size(3);

    OP_REQUIRES(context, images.dims() == 4,
                errors::InvalidArgument("The rank of the images should be 4"));
    OP_REQUIRES(
        context, keypoints.dims() == 4,
        errors::InvalidArgument("The rank of the keypoints tensor should be 4"));
    OP_REQUIRES(context, images.dim_size(0) == keypoints.dim_size(0),
                errors::InvalidArgument("The batch sizes should be the same"));

    OP_REQUIRES(
        context, depth == 4 || depth == 1 || depth == 3,
        errors::InvalidArgument("Channel depth should be either 1 (GRY), "
                                "3 (RGB), or 4 (RGBA)"));

    const int64 batch_size = images.dim_size(0);
    const int64 height = images.dim_size(1);
    const int64 width = images.dim_size(2);
    const int64 color_table_length = 14;

    // 0: yellow
    // 1: blue
    // 2: red
    // 3: lime
    // 4: purple
    // 5: olive
    // 6: maroon
    // 7: navy blue
    // 8: aqua
    // 9: fuchsia
    float color_table[color_table_length][4] = {
        {1, 1, 0, 1},     {0, 0, 1, 1},     {1, 0, 0, 1},   {0, 1, 0, 1},
        {0.5, 0, 0.5, 1}, {0.5, 0.5, 0, 1}, {0.5, 0, 0, 1}, {0, 0, 0.5, 1},
        {0, 1, 1, 1},     {1, 0, 1, 1},
        //10 11 12 13
        {0.8, 0, 0.8, 1}, {0.8, 0.8, 0, 1}, {0.8, 0, 0, 1}, {0, 0, 0.8, 1},
    };
    // Reset first color channel to 1 if image is GRY.
    // For GRY images, this means all bounding boxes will be white.
    if (depth == 1) {
      for (int64 i = 0; i < color_table_length; i++) {
        color_table[i][0] = 1;
      }
    }
    Tensor* output;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({batch_size, height, width, depth}), &output));

    output->tensor<float, 4>() = images.tensor<float, 4>();
    auto canvas = output->tensor<float, 4>();

    for(int64 b = 0; b < batch_size; ++b){
      const int64 num_human = keypoints.dim_size(1);
      const auto tpoints = keypoints.tensor<float, 4>();
      for(int64 h = 0; h < num_human; ++h){
        const int64 num_points = keypoints.dim_size(2);
        for(int64 p = 0; p < num_points; ++p){
          const int64 visible = static_cast<int64>(tpoints(b, h, p, 2));
          if (visible != 1)
            continue;
          int64 color_index = p % color_table_length;
          const int64 prow = static_cast<float>(tpoints(b, h, p, 1)) * (height - 1);
          const int64 min_row_clamp = std::max<int64>(prow - 5 , 0);
          const int64 max_row_clamp = std::min<int64>(prow + 5 , height - 1);
          const int64 pcol = static_cast<float>(tpoints(b, h, p, 0)) * (width - 1);
          const int64 min_col_clamp = std::max<int64>(pcol - 5 , 0);
          const int64 max_col_clamp = std::min<int64>(pcol + 5 , width - 1);
          
          if (prow >= height || prow < 0 ||
              pcol >= width || pcol < 0) {
            LOG(WARNING) << "Keypoints (" << prow
                        << "," << pcol
                        << ") is completely outside the image"
                        << " and will not be drawn.";
            continue;
          }

          CHECK_GE(min_row_clamp, 0);
          CHECK_GE(max_row_clamp, 0);
          CHECK_LT(min_row_clamp, height);
          CHECK_LT(max_row_clamp, height);
          CHECK_GE(min_col_clamp, 0);
          CHECK_GE(max_col_clamp, 0);
          CHECK_LT(min_col_clamp, width);
          CHECK_LT(max_col_clamp, width);

          CHECK_LT(prow, height);
          CHECK_GE(prow, 0);
          CHECK_LT(pcol, width);
          CHECK_GE(pcol, 0);

          if(min_row_clamp >= 0 && max_row_clamp < height &&
            min_col_clamp >= 0 && max_col_clamp < width){
            for (int64 i = min_row_clamp; i <= max_row_clamp; ++i){
              for (int64 j = min_col_clamp; j <= max_col_clamp;++j){
                for (int64 c = 0; c < depth; c++)
                  canvas(b, i, j, c) = static_cast<float>(color_table[color_index][c]);
              }
            }
          }

        }
      }
    }
  }
};


REGISTER_KERNEL_BUILDER(Name("DrawKeypoints").Device(DEVICE_CPU), DrawKeypointsOp);
