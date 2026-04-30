use burn::backend::Cuda;
use burn::tensor::Tensor;

type B = Cuda;

fn main() {
    println!("=== CUDA 버전 테스트 ===");
    println!("CUDA 디바이스 초기화 중...");

    let device = burn::backend::cuda::CudaDevice::new(0);
    println!("✅ 디바이스: {:?}", device);

    let a: Tensor<B, 2> = Tensor::random(
        [4096, 4096],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let b: Tensor<B, 2> = Tensor::random(
        [4096, 4096],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    println!("🔥 풀로드 시작...");
    for i in 0..500 {
        let c = a.clone().matmul(b.clone());
        let _ = c.slice([0..1, 0..1]).into_scalar();
        if i % 50 == 0 {
            println!("  반복 {}/500", i);
        }
    }
    println!("✅ 완료!");
}
