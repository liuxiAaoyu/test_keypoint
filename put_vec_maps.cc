#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"


using namespace tensorflow;

REGISTER_OP("PutVecMaps")
    .Input("images: float32")
    .Input("keypoints: float32")
    .Input("areafactors: float32")
    .Output("vecmaps: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      ::tensorflow::shape_inference::ShapeHandle output;
      
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, -1,  c->MakeDim(::tensorflow::shape_inference::DimensionOrConstant(28)) , &output));

      c->set_output(0, output);
      return Status::OK();
    });

    

class PutVecMapsOp : public OpKernel {
 public:
  explicit PutVecMapsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& images = context->input(0);
    const Tensor& keypoints = context->input(1);
    const Tensor& areafactors = context->input(2);

    OP_REQUIRES(context, images.dims() == 4,
                errors::InvalidArgument("The rank of the images should be 4"));
    OP_REQUIRES(
        context, keypoints.dims() == 4,
        errors::InvalidArgument("The rank of the keypoints tensor should be 4"));
    OP_REQUIRES(
        context, areafactors.dims() == 3,
        errors::InvalidArgument("The rank of the areafactors tensor should be 3"));
    OP_REQUIRES(context, images.dim_size(0) == keypoints.dim_size(0),
                errors::InvalidArgument("The batch sizes should be the same"));
    OP_REQUIRES(context, areafactors.dim_size(0) == keypoints.dim_size(0),
                errors::InvalidArgument("The batch sizes should be the same"));
    OP_REQUIRES(context, areafactors.dim_size(1) == keypoints.dim_size(1),
                errors::InvalidArgument("The human number should be the same"));


    const int64 batch_size = images.dim_size(0);
    const int64 height = images.dim_size(1);
    const int64 width = images.dim_size(2);
    const int64 depth = 28;

    Tensor* output;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({batch_size, height, width, depth}), &output));

    auto canvas = output->tensor<float, 4>();
    const auto tpoints = keypoints.tensor<float, 4>();
    const auto factors = areafactors.tensor<float, 3>();
    const int mid_1[14] = {12, 13, 0, 1, 13, 3, 4, 0, 6, 7, 3, 9, 10, 6};
    const int mid_2[14] = {13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 9};

    for(int64 b = 0; b < batch_size; ++b){
      std::vector<float> thre_v;
      const int64 num_human = keypoints.dim_size(1);
      for(int64 h = 0; h < num_human; ++h){
          float temp = static_cast<float>(factors(b, h, 0));
          if(temp>1)
            thre_v.push_back(16.0);
          else if(temp<=0.5)
            thre_v.push_back(8.0);
          else
            thre_v.push_back(8.0+(temp-0.5)*8.0);
      }
      for(int64 p = 0; p < 14; ++p){
        std::vector<std::vector<float>> p_count(height, std::vector<float> (width, 0)); 
        
        for(int64 h = 0; h < num_human; ++h){
          const int64 pa = mid_1[p];
          const int64 pb = mid_2[p];
          const float visible_a = static_cast<float>(tpoints(b, h, pa, 2));
          const float visible_b = static_cast<float>(tpoints(b, h, pb, 2));
          if (visible_a != 1.0)
            continue;
          if (visible_b != 1.0)
            continue;

          const int64 prow_a = static_cast<float>(tpoints(b, h, pa, 1)) * (height - 1);
          const int64 pcol_a = static_cast<float>(tpoints(b, h, pa, 0)) * (width - 1);
          const int64 prow_b = static_cast<float>(tpoints(b, h, pb, 1)) * (height - 1);
          const int64 pcol_b = static_cast<float>(tpoints(b, h, pb, 0)) * (width - 1);
          float thre = thre_v[h];

          if (prow_a >= height || prow_a < 0 ||
              pcol_a >= width || pcol_a < 0) {
            LOG(WARNING) << "Keypoints (" << prow_a
                        << "," << pcol_a
                        << ") is completely outside the image"
                        << " and will not be drawn.";
            continue;
          }
          if (prow_b >= height || prow_b < 0 ||
              pcol_b >= width || pcol_b < 0) {
            LOG(WARNING) << "Keypoints (" << prow_b
                        << "," << pcol_b
                        << ") is completely outside the image"
                        << " and will not be drawn.";
            continue;
          }

          CHECK_LT(prow_a, height);
          CHECK_GE(prow_a, 0);
          CHECK_LT(pcol_a, width);
          CHECK_GE(pcol_a, 0);
          CHECK_LT(prow_b, height);
          CHECK_GE(prow_b, 0);
          CHECK_LT(pcol_b, width);
          CHECK_GE(pcol_b, 0);

          const int64 min_col = std::max( int(round(std::min(pcol_a, pcol_b)-thre)), 0);
          const int64 max_col = std::min( int(round(std::max(pcol_a, pcol_b)+thre)), (int)width);

          const int64 min_row = std::max( int(round(std::min(prow_a, prow_b)-thre)), 0);
          const int64 max_row = std::min( int(round(std::max(prow_a, prow_b)+thre)), (int)height);

          float norm = sqrt((pcol_b-pcol_a)*(pcol_b-pcol_a) + (prow_b-prow_a)*(prow_b-prow_a));
          float col_factor = (pcol_b-pcol_a) / norm;
          float row_factor = (prow_b-prow_a) / norm;

          for (int64 i = min_row; i < max_row; ++i){
            for (int64 j = min_col; j < max_col; ++j){
              float v_col = j - pcol_a;
              float v_row = i - prow_a;
              float dist = std::abs(v_col*row_factor - v_row*col_factor);

              if(dist < thre){
                float t_count = p_count[i][j];
                if(t_count == 0){
                    canvas(b, i, j, p * 2) = col_factor;// v_col;
                    canvas(b, i, j, p * 2 + 1) = row_factor;//v_row;
                }
                else{
                    canvas(b, i, j, p * 2) = (canvas(b, i, j, p * 2) + col_factor) / (t_count + 1);
                    canvas(b, i, j, p * 2 + 1) = (canvas(b, i, j, p * 2 + 1) + row_factor) / (t_count + 1);
                    p_count[i][j] = t_count + 1;
                }

              }
            }
          }
        

        }
      }
    }
  }
};


REGISTER_KERNEL_BUILDER(Name("PutVecMaps").Device(DEVICE_CPU), PutVecMapsOp);