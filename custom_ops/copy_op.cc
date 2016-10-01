
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <stdio.h>

namespace tensorflow {



typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template<typename Device, typename T>
class MyCopyOp: public OpKernel {
public:
    explicit MyCopyOp(OpKernelConstruction* context) :
            OpKernel(context) {
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        auto in_flat = input.flat<T>();

        printf("Debug MyCopyOp Features: %s \n",input.DebugString().c_str());

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                context->allocate_output(0, input.shape(), &output));

        auto out_flat = output->flat<T>();
        out_flat.setZero();

        for (int d = 0; d < input.dims(); ++d) {
            for (int i = 0; i < input.dim_size(d); ++i) {
                out_flat(d * input.dim_size(d) + i) = in_flat(
                        d * input.dim_size(d) + i);
            }
        }

        printf("Debug MyCopyOp Output: %s \n",output->DebugString().c_str());
    }

};


template<typename Device, typename T>
class MyCopyGradOp: public OpKernel {
public:
    explicit MyCopyGradOp(OpKernelConstruction* context) :
            OpKernel(context) {

    }

    void Compute(OpKernelContext* context) override {
        printf("called MyCopyGradOp.Compute() \n");
        const Tensor& gradients = context->input(0);
        const Tensor& features = context->input(1);
        printf("Debug MyCopyOpGrad Gradients: %s \n",gradients.DebugString().c_str());
        printf("Debug MyCopyOpGrad Features: %s \n",features.DebugString().c_str());

        TensorShape output_shape = features.shape();

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                context->allocate_output(0, output_shape, &output));
        output->flat<T>().setZero();

        const T* btm_ptr = gradients.flat<T>().data();
        T* top_ptr = output->flat<T>().data();

        for (int i = 0; i < gradients.NumElements(); ++i) {
            top_ptr[i] = btm_ptr[i];
        }

        printf("Debug MyCopyOpGrad Output: %s \n",output->DebugString().c_str());
        printf("---------------------------------- \n");
    }

};


REGISTER_OP("MyCopy")
.Input("features: T")
.Output("output: T")
.Attr("T: realnumbertype")
.Doc(R"doc(
Copies all input values to the output
)doc");

REGISTER_OP("MyCopyGrad")
.Input("gradients: T")
.Input("features: T")
.Output("backprops: T")
.Attr("T: realnumbertype")
.Doc(R"doc(
TODO!!
)doc");


#define REGISTER_MYCOPY_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(                                                      \
      Name("MyCopy").Device(DEVICE_CPU).TypeConstraint<type>("T"),              \
      MyCopyOp<Eigen::ThreadPoolDevice, type>);                                 \
  REGISTER_KERNEL_BUILDER(                                                      \
      Name("MyCopyGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),          \
      MyCopyGradOp<Eigen::ThreadPoolDevice, type>);                             //  \
  // REGISTER_KERNEL_BUILDER(                                                      \
  //     Name("MyCopy").Device(DEVICE_GPU).TypeConstraint<type>("T"),              \
  //     MyCopyOp<Eigen::GpuDevice, type>);                                        \
  // REGISTER_KERNEL_BUILDER(                                                      \
  //     Name("MyCopyGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),          \
  //     MyCopyGradOp<Eigen::GpuDevice, type>);                                


REGISTER_MYCOPY_KERNELS(float); 
REGISTER_MYCOPY_KERNELS(int);
REGISTER_MYCOPY_KERNELS(double);


}


