use burn::backend::Wgpu;
use burn::tensor::Tensor;

type B = Wgpu;

fn main() {
    println!("=== wgpu 버전 테스트 (Ollama 방식) ===");
    println!("GPU 자동 감지 중... (CUDA 불필요!)");

    let device = burn::backend::wgpu::WgpuDevice::default();
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
    println!("✅ 완료! CUDA 없이 GPU 풀로드 성공!");
}
