#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"


using namespace tensorflow;

REGISTER_OP("PutGaussianMaps")
    .Input("images: float32")
    .Input("keypoints: float32")
    .Output("gaussianmaps: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      ::tensorflow::shape_inference::ShapeHandle output;
      
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, -1,  c->MakeDim(::tensorflow::shape_inference::DimensionOrConstant(14)) , &output));

      c->set_output(0, output);
      return Status::OK();
    });

    

class PutGaussianMapsOp : public OpKernel {
 public:
  explicit PutGaussianMapsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& images = context->input(0);
    const Tensor& keypoints = context->input(1);

    OP_REQUIRES(context, images.dims() == 4,
                errors::InvalidArgument("The rank of the images should be 4"));
    OP_REQUIRES(
        context, keypoints.dims() == 4,
        errors::InvalidArgument("The rank of the keypoints tensor should be 4"));
    OP_REQUIRES(context, images.dim_size(0) == keypoints.dim_size(0),
                errors::InvalidArgument("The batch sizes should be the same"));


    const int64 batch_size = images.dim_size(0);
    const int64 height = images.dim_size(1);
    const int64 width = images.dim_size(2);
    const int64 depth = 14;

    Tensor* output;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({batch_size, height, width, depth}), &output));

    auto canvas = output->tensor<float, 4>();

    for(int64 b = 0; b < batch_size; ++b){
      const int64 num_points = keypoints.dim_size(2);
      for(int64 p = 0; p < num_points; ++p){
        const int64 num_human = keypoints.dim_size(1);
        const auto tpoints = keypoints.tensor<float, 4>();
        for(int64 h = 0; h < num_human; ++h){

          const int64 visible = static_cast<int64>(tpoints(b, h, p, 2));
          if (visible != 1)
            continue;

          const int64 prow = static_cast<float>(tpoints(b, h, p, 1)) * (height - 1);
          const int64 pcol = static_cast<float>(tpoints(b, h, p, 0)) * (width - 1);
          float sigma = 17.0;
                              
          if (prow >= height || prow < 0 ||
              pcol >= width || pcol < 0) {
            LOG(WARNING) << "Keypoints (" << prow
                        << "," << pcol
                        << ") is completely outside the image"
                        << " and will not be drawn.";
            continue;
          }

          CHECK_LT(prow, height);
          CHECK_GE(prow, 0);
          CHECK_LT(pcol, width);
          CHECK_GE(pcol, 0);

          for (int64 i = 0; i < height; ++i){
            for (int64 j = 0; j < width;++j){
              float dist = (i-prow)*(i-prow) + (j-pcol)* (j-pcol);
              float exponent = dist / 2.0 / sigma / sigma;
              if(exponent > 4.6052) //ln(100) = -ln(1%)
                continue;
              canvas(b, i, j, p) += exp(-exponent);
              if(canvas(b, i, j, p) > 1)
                canvas(b, i, j, p) = 1;
            }
          }
        

        }
      }
    }
  }
};


REGISTER_KERNEL_BUILDER(Name("PutGaussianMaps").Device(DEVICE_CPU), PutGaussianMapsOp);
